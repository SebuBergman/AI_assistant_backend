from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import datetime
import os
import uvicorn
import json
import boto3

# Your existing imports
from email_assistant import rewrite_email_stream, EmailRequest
from ai_assistant import ask_ai, AI_Request
from tools import is_tool_supported

# RAG-related imports
from pymilvus import MilvusClient
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db import (
    QUERY_CACHE_COLLECTION,
    delete_document_embeddings,
    delete_pdf_metadata,
    get_pdf_metadata,
    insert_pdf_metadata,
    clear_cache_entries,
    get_cache_stats as db_get_cache_stats,
    store_query_result,
    find_similar_cached_query,
    clear_all_embeddings,
    clear_all_pdfs,
    get_milvus_collection_stats,
)
from S3_bucket import delete_s3_file, upload_to_s3, delete_all_s3_files
from rag_search import (
    vector_search,
    keyword_search,
    hybrid_search,
)
from vectorstore_manager import (
    get_vectorstore, 
    embeddings, 
    COLLECTION_NAME, 
    MILVUS_CONNECTION, 
    vectorstore,
    reset_vectorstore
)

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# AI client initialization
openai_client = OpenAI()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# RAG Configuration
UPLOAD_PATH = "./data"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Zilliz Cloud / Milvus config
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")

# Initialize Milvus client for Zilliz Cloud
milvus_client = MilvusClient(
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_TOKEN
)

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# CORS configuration - Combined both origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class ChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 20240

class QueryRequest(BaseModel):
    question: str
    keyword: str = ""
    file_name: str = ""
    cached: bool = False
    alpha: float = 0.7

# Extended AI_Request to include RAG options
class ExtendedAI_Request(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 20240
    ragEnabled: bool = False
    file_name: Optional[str] = ""
    keyword: Optional[str] = ""
    cached: Optional[bool] = False
    alpha: Optional[float] = 0.7

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_rag_context(question: str, file_name: str = "", keyword: str = "", 
                    cached: bool = False, alpha: float = 0.7):
    """Get RAG context for a query"""
    try:
        vs = get_vectorstore()
        if vs is None:
            return None, "No documents available"
        
        # Check cache if enabled
        if cached:
            query_embedding = embeddings.embed_query(question)
            cached_result = find_similar_cached_query(
                query_embedding=query_embedding,
                threshold=0.85
            )
            if cached_result:
                return cached_result.get("context", ""), "cached"
        
        # Perform vector search
        vector_results = vector_search(
            query=question,
            file_name=file_name if file_name else None,
            limit=7
        )
        
        # Perform keyword search if keyword provided
        keyword_results = []
        if keyword:
            keyword_results = keyword_search(
                query=keyword,
                file_name=file_name if file_name else None,
                limit=5
            )
        
        # Combine with hybrid search
        if keyword_results:
            final_results = hybrid_search(
                vector_results=vector_results,
                keyword_results=keyword_results,
                alpha=alpha,
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
                    f"Document {i+1} (Hybrid: {result['hybrid_score']:.3f})\n"
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
        return context, search_method
    
    except Exception as e:
        print(f"Error getting RAG context: {e}")
        return None, f"Error: {str(e)}"

# ============================================================================
# ENDPOINTS - AI Assistant
# ============================================================================

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Assistant API with RAG Support!"}

@app.post("/email_assistant")
async def email_assistant_endpoint(request: EmailRequest):
    try:
        return await rewrite_email_stream(request)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def ask_ai_endpoint(request: ExtendedAI_Request):
    """Unified streaming generator with optional RAG support"""
    
    async def generate():
        try:
            # Prepare the prompt
            prompt = request.prompt
            
            # If RAG is enabled, fetch context and augment prompt
            if request.ragEnabled:
                context, search_method = get_rag_context(
                    question=request.prompt,
                    file_name=request.file_name or "",
                    keyword=request.keyword or "",
                    cached=request.cached or False,
                    alpha=request.alpha or 0.7
                )
                
                if context:
                    # Augment the prompt with RAG context
                    prompt = f"""Use the following context to help answer the question.
                    If the context is relevant, use it. If not, answer based on your knowledge.

                    Context:
                    {context}

                    Question: {request.prompt}

                    Answer:"""
                    # Send a metadata message about RAG being used
                    yield f"data: {json.dumps({'metadata': {'rag_enabled': True, 'search_method': search_method}})}\n\n"
            
            # Create a modified request with the augmented prompt
            ai_request = AI_Request(
                question=prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Get the streaming generator from ask_ai
            stream = ask_ai(
                ai_request,
                openai_client,
                deepseek_client,
                anthropic_client
            )

            # Stream the content with SSE formatting
            for chunk in stream:
                if isinstance(chunk, str):
                    if chunk.startswith('{'):
                        yield f"data: {chunk}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': str(chunk)})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    
@app.get("/api/models")
async def list_models():
    """Get list of available models"""
    from ai_assistant import MODEL_FUNCTIONS
    return {"models": list(MODEL_FUNCTIONS.keys())}

@app.get("/api/tools/{model_name}")
async def check_tool_support(model_name: str):
    """Check if a specific model supports tool calling"""
    return {
        "model": model_name,
        "supports_tools": is_tool_supported(model_name)
    }

# ============================================================================
# ENDPOINTS - RAG System
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint to verify connections"""
    try:
        vs = get_vectorstore()
        milvus_status = "connected" if vs is not None else "disconnected"
        stats = get_milvus_collection_stats()
        
        return {
            "status": "healthy",
            "milvus": milvus_status,
            "collection": COLLECTION_NAME,
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF"""
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Upload to S3
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"fetch_pdfs/{file.filename}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)
        
        # Load and process PDF with LangChain
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["file_name"] = file.filename
            doc.metadata["source"] = s3_url
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Get or initialize vectorstore
        vs = get_vectorstore()
        
        if vs is None:
            print("Initializing vectorstore connection...")
            import vectorstore_manager
            
            vs = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION,
                consistency_level="Strong",
                drop_old=False,
                auto_id=True,
            )
            vectorstore_manager.vectorstore = vs
            print("✓ Vectorstore connected to existing collection")
        
        # Add documents to vectorstore
        print(f"Adding {len(splits)} documents to vectorstore...")
        vs.add_documents(splits)
        
        # Store PDF metadata in Milvus
        insert_pdf_metadata(file.filename, s3_url)
        
        # Clean up local file
        os.remove(file_path)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "s3_url": s3_url,
            "chunks_created": len(splits),
            "collection": COLLECTION_NAME
        }
    
    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/fetch_pdfs")
def get_pdfs():
    """Get the list of available PDFs."""
    try:
        pdfs = get_pdf_metadata()
        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDFs: {str(e)}")

@app.post("/query")
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
            "sources": final_results,
            "context": context,
            "from_cache": False
        }
    
    except Exception as e:
        print(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS - Cache Management
# ============================================================================

@app.get("/cache/stats")
def cache_stats_endpoint():
    """Endpoint to return cache stats."""
    try:
        stats = db_get_cache_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear_old")
def clear_old_cache_entries_endpoint(days: int = 30):
    """Clear cache entries older than specified days (default: 30)."""
    try:
        if days <= 0:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be positive"
            )
            
        deleted_count = clear_cache_entries(days=days)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} entries older than {days} days",
            "deleted_count": deleted_count,
            "days": days
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing old cache entries: {str(e)}"
        )

@app.post("/cache/clear_all")
def clear_whole_cache():
    """Clear ALL cache entries."""
    try:
        deleted_count = clear_cache_entries()
        return {
            "status": "success",
            "message": f"Deleted all {deleted_count} cache entries",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )

@app.get("/cache/entries")
def list_cache_entries(limit: int = 10):
    """List recent cached queries from Milvus."""
    try:
        results = milvus_client.query(
            collection_name=QUERY_CACHE_COLLECTION,
            filter="pk >= 0",
            output_fields=["query", "answer", "timestamp"],
            limit=limit
        )

        results = sorted(results, key=lambda x: x["timestamp"], reverse=True)

        for r in results:
            r["timestamp"] = datetime.datetime.fromtimestamp(r["timestamp"]).isoformat()

        return {"entries": results, "count": len(results)}

    except Exception as e:
        print(f"Error listing cache entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS - Data Management
# ============================================================================

@app.post("/delete_document")
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
   
@app.post("/clear_all")
async def clear_all():
    """Clear all data (embeddings, PDFs, and S3)"""
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

@app.get("/milvus/stats")
def milvus_stats():
    """Get Milvus/Zilliz collection statistics"""
    try:
        stats = get_milvus_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/milvus/schema")
def get_collection_schema():
    """Get the current collection schema to verify it's correct"""
    try:
        from db import milvus_client, MILVUS_COLLECTION_NAME
        
        if MILVUS_COLLECTION_NAME not in milvus_client.list_collections():
            return {
                "error": f"Collection '{MILVUS_COLLECTION_NAME}' does not exist",
                "collections": milvus_client.list_collections()
            }
        
        schema_info = milvus_client.describe_collection(MILVUS_COLLECTION_NAME)
        fields = schema_info.get('fields', [])
        field_names = [f.get('name') for f in fields]
        has_correct_schema = all(name in field_names for name in ['pk', 'vector', 'text'])
        
        return {
            "collection": MILVUS_COLLECTION_NAME,
            "schema": schema_info,
            "field_names": field_names,
            "has_correct_schema": has_correct_schema,
            "expected_fields": ["pk", "vector", "text"],
            "recommendation": "Schema looks good!" if has_correct_schema else 
                            "⚠️ Schema incorrect! Run POST /clear_all to recreate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================
    
def main():
    """Main function to run the FastAPI application with uvicorn server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
if __name__ == "__main__":
    main()