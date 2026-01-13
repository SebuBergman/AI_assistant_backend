import re

from typing import List, Dict
from rank_bm25 import BM25Okapi

from app.db.vectorstore_manager import get_vectorstore
from app.db.database import milvus_client, MILVUS_COLLECTION_NAME

_WORD = re.compile(r"\w+")

def _tokenize(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())

def _collection_has_field(field_name: str) -> bool:
    try:
        if MILVUS_COLLECTION_NAME not in milvus_client.list_collections():
            return False
        schema_info = milvus_client.describe_collection(MILVUS_COLLECTION_NAME)
        fields = [f.get("name") for f in schema_info.get("fields", [])]
        return field_name in fields
    except Exception as e:
        print(f"Could not inspect collection schema: {e}")
        # Be conservative: return False so we fallback to client filtering
        return False

def keyword_search(query: str, file_name: str = None, limit: int = 7) -> List[Dict]:
    """
        Keyword search using BM25 over a retrieved candidate set.
        Returns normalized keyword scores in [0, 1] so hybrid mixing is stable.
    """
    try:
        vs = get_vectorstore()
        if vs is None:
            return []

        # Pull a larger candidate set (tune this)
        candidates = vs.similarity_search(query=" ", k=300)

        # Optional file filter (client-side)
        if file_name:
            candidates = [d for d in candidates if d.metadata.get("file_name", "") == file_name]

        if not candidates:
            return []

        tokenized_docs = [_tokenize(d.page_content) for d in candidates]
        bm25 = BM25Okapi(tokenized_docs)

        q_tokens = _tokenize(query)
        raw_scores = bm25.get_scores(q_tokens)  # higher is better

        # Normalize BM25 scores to [0,1] for hybrid
        max_s = float(max(raw_scores)) if len(raw_scores) else 0.0
        min_s = float(min(raw_scores)) if len(raw_scores) else 0.0
        denom = (max_s - min_s) if max_s != min_s else 1.0
        norm_scores = [(float(s) - min_s) / denom for s in raw_scores]

        # Rank
        ranked = sorted(
            zip(candidates, norm_scores),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        results = []
        for doc, score in ranked:
            results.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "text": doc.page_content,
                "file_name": doc.metadata.get("file_name", ""),
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "score": float(score),
                "search_type": "keyword",
            })

        return results

    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

def vector_search(query: str, file_name: str = None, limit: int = 7) -> List[Dict]:
    """Perform semantic vector search. If file_name is provided and exists in the collection schema,
       use server-side expr filter for efficiency. Otherwise fallback to client-side filtering."""
    try:
        vs = get_vectorstore()
        if vs is None:
            return []

        search_kwargs = {"k": limit}
        server_side_filter = False
        if file_name:
            if _collection_has_field("file_name"):
                server_side_filter = True
                # Milvus expr expects quotes around strings; escape if needed
                safe_name = file_name.replace('"', '\\"')
                search_kwargs["expr"] = f'file_name == "{safe_name}"'
            else:
                print("Server-side 'file_name' field not present — will filter client-side instead.")

        docs_with_scores = vs.similarity_search_with_score(query, **search_kwargs)

        # If server-side filtering wasn't possible but file_name was requested, we still need to filter results
        results = []
        for doc, score in docs_with_scores:
            meta_file = doc.metadata.get("file_name", "")
            if file_name and not server_side_filter:
                # Skip entries that don't match requested file_name
                if meta_file != file_name:
                    continue

            # Convert COSINE distance [0,2] to similarity score [0,1]
            # where 0 distance = 1.0 similarity (identical)
            # and 2 distance = 0.0 similarity (opposite)
            similarity_score = 1 - (score / 2)

            results.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "text": doc.page_content,
                "file_name": meta_file,
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "score": float(similarity_score),
                "search_type": "vector"
            })

        return results

    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def hybrid_search(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    alpha: float = 0.7,
    limit: int = 7,
    use_rrf: bool = True,
    k: int = 60
) -> List[Dict]:
    """
    Combine vector and keyword results with weighted scoring.
    
    Args:
        vector_results: Results from vector search
        keyword_results: Results from keyword search
        alpha: Weight for vector scores (0-1). Higher = more weight on vectors
        limit: Maximum number of results to return
        use_rrf: Use Reciprocal Rank Fusion instead of score normalization
        k: RRF constant (default 60)
    """
    
    if use_rrf:
        return _hybrid_search_rrf(vector_results, keyword_results, alpha, limit, k)
    else:
        return _hybrid_search_normalized(vector_results, keyword_results, alpha, limit)


def _hybrid_search_rrf(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    alpha: float,
    limit: int,
    k: int = 60
) -> List[Dict]:
    """Reciprocal Rank Fusion - rank-based merging (recommended)"""
    
    # Create rank maps (position-based scoring)
    vector_ranks = {r["chunk_id"]: idx for idx, r in enumerate(vector_results)}
    keyword_ranks = {r["chunk_id"]: idx for idx, r in enumerate(keyword_results)}
    
    # Create lookup maps for metadata
    vector_map = {r["chunk_id"]: r for r in vector_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}
    
    # Get all unique chunks
    all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
    
    merged = []
    for chunk_id in all_chunk_ids:
        # Calculate RRF scores
        vector_rrf = 1 / (k + vector_ranks[chunk_id] + 1) if chunk_id in vector_ranks else 0
        keyword_rrf = 1 / (k + keyword_ranks[chunk_id] + 1) if chunk_id in keyword_ranks else 0
        
        # Weighted combination
        hybrid_score = (alpha * vector_rrf) + ((1 - alpha) * keyword_rrf)
        
        # Get metadata from whichever source has it
        source_data = vector_map.get(chunk_id) or keyword_map.get(chunk_id)
        
        # Get original scores for debugging
        vector_score = vector_map[chunk_id]["score"] if chunk_id in vector_map else 0
        keyword_score = keyword_map[chunk_id]["score"] if chunk_id in keyword_map else 0
        
        print(f"RRF scores for chunk {chunk_id}: "
              f"v_rank={vector_ranks.get(chunk_id, 'N/A')} (score={vector_score:.3f}), "
              f"k_rank={keyword_ranks.get(chunk_id, 'N/A')} (score={keyword_score:.3f}), "
              f"hybrid={hybrid_score:.4f}")
        
        merged.append({
            "chunk_id": chunk_id,
            "chunk_index": source_data.get("chunk_index"),
            "text": source_data["text"],
            "file_name": source_data["file_name"],
            "page": source_data.get("page"),
            "source": source_data.get("source"),
            "score": hybrid_score,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score,
            "vector_rank": vector_ranks.get(chunk_id),
            "keyword_rank": keyword_ranks.get(chunk_id),
            "search_type": "hybrid_rrf"
        })
    
    # Sort by hybrid score descending
    merged_sorted = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
    return merged_sorted[:limit]


def _hybrid_search_normalized(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    alpha: float,
    limit: int
) -> List[Dict]:
    """Score normalization approach - normalize each score set to [0,1]"""
    
    # Normalize scores to [0, 1] range
    def normalize_scores(results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            return {r["chunk_id"]: 1.0 for r in results}
        
        return {
            r["chunk_id"]: (r["score"] - min_score) / score_range 
            for r in results
        }
    
    # Normalize both result sets
    vector_normalized = normalize_scores(vector_results)
    keyword_normalized = normalize_scores(keyword_results)
    
    # Create lookup maps
    vector_map = {r["chunk_id"]: r for r in vector_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}
    
    # Get all unique chunks
    all_chunk_ids = set(vector_map.keys()) | set(keyword_map.keys())
    
    merged = []
    for chunk_id in all_chunk_ids:
        # Use normalized scores
        vector_score_norm = vector_normalized.get(chunk_id, 0)
        keyword_score_norm = keyword_normalized.get(chunk_id, 0)
        
        hybrid_score = (alpha * vector_score_norm) + ((1 - alpha) * keyword_score_norm)
        
        # Get metadata
        source_data = vector_map.get(chunk_id) or keyword_map.get(chunk_id)
        
        # Original scores for reference
        vector_score_orig = vector_map[chunk_id]["score"] if chunk_id in vector_map else 0
        keyword_score_orig = keyword_map[chunk_id]["score"] if chunk_id in keyword_map else 0
        
        print(f"Normalized scores for chunk {chunk_id}: "
              f"vector={vector_score_orig:.3f}→{vector_score_norm:.3f}, "
              f"keyword={keyword_score_orig:.3f}→{keyword_score_norm:.3f}, "
              f"hybrid={hybrid_score:.3f}")
        
        merged.append({
            "chunk_id": chunk_id,
            "chunk_index": source_data.get("chunk_index"),
            "text": source_data["text"],
            "file_name": source_data["file_name"],
            "page": source_data.get("page"),
            "source": source_data.get("source"),
            "score": hybrid_score,
            "vector_score": vector_score_orig,
            "keyword_score": keyword_score_orig,
            "hybrid_score": hybrid_score,
            "search_type": "hybrid_normalized"
        })
    
    # Sort by hybrid score descending
    merged_sorted = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
    return merged_sorted[:limit]