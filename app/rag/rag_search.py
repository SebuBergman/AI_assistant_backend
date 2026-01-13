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
    limit: int = 7
) -> List[Dict]:
    """
    Union-style hybrid search:
    - Vector-only results keep vector score
    - Keyword-only results keep keyword score
    - Overlapping results get blended
    """

    vector_map = {r["chunk_id"]: r for r in vector_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}

    all_chunk_ids = set(vector_map) | set(keyword_map)

    merged = []

    for chunk_id in all_chunk_ids:
        vec = vector_map.get(chunk_id)
        kw = keyword_map.get(chunk_id)

        vector_score = vec["score"] if vec else None
        keyword_score = kw["score"] if kw else None

        # Decide hybrid score
        if vector_score is not None and keyword_score is not None:
            hybrid_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
            search_type = "hybrid_both"
        elif vector_score is not None:
            hybrid_score = vector_score
            search_type = "hybrid_vector_only"
        else:
            hybrid_score = keyword_score
            search_type = "hybrid_keyword_only"

        source = vec or kw

        print(
            f"Hybrid score for {chunk_id}: "
            f"vector={vector_score}, keyword={keyword_score}, hybrid={hybrid_score}"
        )

        print(f"Results of RAG search - chunk_id: {chunk_id}, source: {source}")

        merged.append({
            "chunk_id": chunk_id,
            "chunk_index": source.get("chunk_index"),
            "text": source["text"],
            "file_name": source["file_name"],
            "page": source.get("page"),
            "source": source.get("source"),
            "score": hybrid_score,
            "vector_score": vector_score or 0.0,
            "keyword_score": keyword_score or 0.0,
            "hybrid_score": hybrid_score,
            "search_type": search_type,
        })

        print(merged)

    return sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)[:limit]
