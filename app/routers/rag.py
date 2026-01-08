from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.database import (
    delete_document_embeddings,
    delete_pdf_metadata,
    get_pdf_metadata,
    clear_cache_entries,
    store_query_result,
    find_similar_cached_query,
    clear_all_embeddings,
    clear_all_pdfs,
)
from app.db.S3_bucket import delete_s3_file, delete_all_s3_files
from app.rag.rag_search import (
    vector_search,
    keyword_search,
    hybrid_search,
)
from app.db.vectorstore_manager import (
    get_vectorstore, 
    embeddings, 
    reset_vectorstore
)
from app.dependencies import llm

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    keyword: str = ""
    file_name: str = ""
    cached: bool = False
    alpha: float = 0.7

@router.get("/fetch_documents", tags=["RAG"])
async def fetch_documents():
    """
    Fetch all uploaded PDFs with optional chunk data from Milvus
    """
    try:
        # 1️⃣ Fetch PDF metadata
        documents = get_pdf_metadata()  # should return List[Dict]
        if not documents:
            return {"documents": []}

        vs = get_vectorstore()  # Milvus vectorstore
        results = []

        for doc in documents:
            # Default empty chunks
            chunks: List[Dict[str, Any]] = []

            # 2️⃣ Fetch chunk data if vectorstore exists and has data
            if vs:
                try:
                    # Access the underlying Milvus client
                    milvus_client = vs.col  # or vs.client depending on your LangChain version
                    
                    # Use async query if your vs is AsyncMilvusClient
                    vector_results = milvus_client.query(
                        expr=f'file_id == "{doc["file_id"]}"',
                        output_fields=["chunk_id", "chunk_index", "page", "text"]
                    )
                    
                    for r in vector_results:
                        chunks.append({
                            "chunk_id": r.get("chunk_id"),
                            "chunk_index": r.get("chunk_index"),
                            "page": r.get("page"),
                            "content": r.get("text"),
                        })
                except Exception as e:
                    print(f"Warning: Failed to fetch chunks for {doc['file_name']}: {e}")

            # 3️⃣ Build final document object
            results.append({
                "file_name": doc["file_name"],
                "file_path": doc.get("file_path") or doc.get("source"),
                "upload_date": doc.get("upload_date") or doc.get("timestamp"),
                "file_size": doc.get("file_size", 0),
                "file_id": doc["file_id"],
                "chunks": chunks,
            })

        return {"documents": results}

    except Exception as e:
        print(f"Error in fetch_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", tags=["RAG"])
async def query(request: QueryRequest):
    """Query the RAG system with custom Milvus caching"""
    try:
        print(f"Query: {request.question}, Keyword: {request.keyword}, File: {request.file_name}, Cached: {request.cached}")
        
        vs = get_vectorstore()
        if vs is None:
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded yet. Please upload a PDF first."
            )
        
        # Check cache if enabled
        cached_result = None
        query_embedding = None
        
        if request.cached:
            query_embedding = embeddings.embed_query(request.question)
            cached_result = find_similar_cached_query(
                query_embedding=query_embedding,
                threshold=0.85
            )
            
            if cached_result:
                print("✓ Using cached result")
                return {
                    "answer": cached_result["answer"],
                    "search_method": "cached",
                    "results_count": 0,
                    "sources": [],
                    "context": cached_result.get("context", ""),
                    "from_cache": True
                }
        
        # Perform vector search
        vector_results = vector_search(
            query=request.question,
            file_name=request.file_name if request.file_name else None,
            limit=7
        )
        
        # Perform keyword search (only if keyword provided)
        keyword_results = []
        if request.keyword:
            keyword_results = keyword_search(
                query=request.keyword,
                file_name=request.file_name if request.file_name else None,
                limit=5
            )
        
        # Combine with hybrid search
        if keyword_results:
            final_results = hybrid_search(
                vector_results=vector_results,
                keyword_results=keyword_results,
                alpha=request.alpha,
                limit=7
            )
            search_method = "hybrid"
        else:
            final_results = vector_results
            search_method = "vector_only"
        
        # Build context from results
        context_lines = []
        for i, result in enumerate(final_results):
            if search_method == "hybrid":
                context_lines.append(
                    f"Document {i+1} (Hybrid: {result['hybrid_score']:.3f} | "
                    f"Vector: {result['vector_score']:.3f} | "
                    f"Keyword: {result['keyword_score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{result['content']}\n"
                )
            else:
                context_lines.append(
                    f"Document {i+1} (Score: {result['score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{result['content']}\n"
                )
        
        context = "\n".join(context_lines)
        
        # Create prompt
        prompt = f"""Use the following context to answer the question.
        If you don't know the answer, say so. Be concise and factual.

        Context:
        {context}

        Question: {request.question}

        Answer:"""
        
        # Get answer from LLM
        response = llm.invoke(prompt)
        answer = response.content
        
        # Store in cache if enabled
        if request.cached and query_embedding is not None:
            store_query_result(
                query_text=request.question,
                query_embedding=query_embedding,
                answer=answer,
                context=context,
                threshold=0.85
            )
        
        return {
            "answer": answer,
            "search_method": search_method,
            "results_count": len(final_results),
            "sources": [
                {
                    "chunk_id": r["chunk_id"],
                    "file_name": r["file_name"],
                    "page": r["page"],
                    "content": r["content"],
                    "score": r["hybrid_score"] if "hybrid_score" in r else r.get("score"),
                    "search_type": r["search_type"],
                    "source": r["source"],
                }
                for r in final_results
            ],
            "from_cache": False
        }
    
    except Exception as e:
        print(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete_document", tags=["RAG"])
async def delete_document(request: dict):
    """Delete a specific document (embeddings, PDF metadata, and S3 file)"""
    try:
        file_name = request.get("file_name")
        if not file_name:
            raise HTTPException(status_code=400, detail="file_name is required")
        
        # Delete embeddings for this document
        embeddings_deleted = delete_document_embeddings(file_name)
        
        # Delete PDF metadata
        pdf_metadata_deleted = delete_pdf_metadata(file_name)
        
        # Delete from S3
        s3_deleted = delete_s3_file(file_name)
        
        # Reset vectorstore to reflect changes
        reset_vectorstore()
        
        return {
            "status": "success",
            "message": f"Document '{file_name}' deleted successfully",
            "embeddings_deleted": embeddings_deleted,
            "pdf_metadata_deleted": pdf_metadata_deleted,
            "s3_file_deleted": s3_deleted
        }
    
    except Exception as e:
        print(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear_all", tags=["RAG"])
async def clear_all():
    """Clear all data (embeddings, PDFs, cache, and S3)"""
    try:
        embeddings_cleared = clear_all_embeddings()
        reset_vectorstore()
        pdfs_deleted = clear_all_pdfs()
        s3_deleted = delete_all_s3_files()
        clear_cache_entries()
        
        return {
            "status": "success",
            "message": "All data cleared",
            "embeddings_cleared": embeddings_cleared,
            "pdf_metadata_deleted": pdfs_deleted,
            "s3_files_deleted": s3_deleted
        }
    
    except Exception as e:
        print(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))