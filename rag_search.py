from typing import List, Dict
from vectorstore_manager import get_vectorstore
from db import milvus_client, MILVUS_COLLECTION_NAME

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
    """Perform keyword search on documents; fallback to client-side filtering if needed."""
    try:
        vs = get_vectorstore()
        if vs is None:
            return []

        # If file_name field exists server-side, we could in principle use expr filtering in the vectorstore query,
        # but keyword search here does a simple content substring scan (LangChain returns metadata)
        # We'll get a reasonably large k then filter:
        docs = vs.similarity_search(query="", k=100)
        query_lower = query.lower()
        results = []
        for doc in docs:
            if query_lower in doc.page_content.lower():
                # If file_name not present in doc.metadata, default to empty string
                results.append({
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content": doc.page_content,
                    "file_name": doc.metadata.get("file_name", ""),
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "score": 1.0,
                    "search_type": "keyword",
                })
                if len(results) >= limit:
                    break
        print(f"Keyword search found {len(results)} results")
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
            results.append({
                "content": doc.page_content,
                "file_name": meta_file,
                "score": float(score),
                "search_type": "vector"
            })

        print(f"Vector search found {len(results)} results (server_filter={server_side_filter})")
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
    """Combine vector and keyword results with weighted scoring"""
    print(f"Combining {len(vector_results)} vector + {len(keyword_results)} keyword results")
    
    # Map keyword results by chunk_id
    keyword_map = {r["chunk_id"]: r["score"] for r in keyword_results}
    
    merged = []
    seen_chunks = set()
    
    # Merge vector results
    for vec in vector_results:
        chunk_id = vec["chunk_id"]
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        vector_score = vec["score"]
        keyword_score = keyword_map.get(chunk_id, 0)
        hybrid_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
        
        merged.append({
            "chunk_id": chunk_id,
            "chunk_index": vec.get("chunk_index"),
            "content": vec["content"],
            "file_name": vec["file_name"],
            "page": vec.get("page"),
            "source": vec.get("source"),
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score,
            "search_type": "hybrid"
        })
    
    # Add keyword-only results
    for kw in keyword_results:
        chunk_id = kw["content"]
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        merged.append({
            "chunk_id": chunk_id,
            "chunk_index": kw.get("chunk_index"),
            "content": kw["content"],
            "file_name": kw["file_name"],
            "page": kw.get("page"),
            "source": kw.get("source"),
            "vector_score": 0,
            "keyword_score": kw["score"],
            "hybrid_score": (1 - alpha) * kw["score"],
            "search_type": "hybrid"
        })
    
    # Sort by hybrid_score descending
    merged_sorted = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
    print(f"Hybrid search returning top {min(limit, len(merged_sorted))} results")
    return merged_sorted[:limit]