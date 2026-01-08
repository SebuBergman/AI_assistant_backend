from fastapi import APIRouter

from app.db.database import get_milvus_collection_stats
from app.db.vectorstore_manager import COLLECTION_NAME, get_vectorstore

router = APIRouter()

@router.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the AI Assistant API with RAG Support!",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat/*",
            "email": "/email_assistant",
            "ai": "/api/*",
            "rag": "/upload, /query, /fetch_pdfs",
            "cache": "/cache/*",
            "milvus": "/milvus/*",
            "management": "/delete_document, /clear_all"
        }
    }

@router.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint to verify connections and database status"""
    from app.services.chat_service import ChatService
    
    try:
        vs = get_vectorstore()
        milvus_status = "connected" if vs is not None else "disconnected"
        stats = get_milvus_collection_stats()
        db_status = await ChatService.test_connection()
        
        return {
            "status": "healthy",
            "milvus": milvus_status,
            "collection": COLLECTION_NAME,
            "stats": stats,
            "database": "connected" if db_status else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }