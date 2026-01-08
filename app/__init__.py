from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils import file_helpers
from .routers import chat, email, rag, cache, milvus
from .config import lifespan, APP_NAME, APP_DESCRIPTION, APP_VERSION
from app.assistants.chat_title import router as chat_title_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Include routers
    app.include_router(chat.router, prefix="/api/chats", tags=["Chats"])
    app.include_router(email.router, prefix="/email", tags=["Email"])
    app.include_router(rag.router, prefix="/rag", tags=["RAG"])
    app.include_router(file_helpers.router, prefix="/files", tags=["Files"])
    app.include_router(cache.router, prefix="/cache", tags=["Cache"])
    app.include_router(milvus.router, prefix="/milvus", tags=["Milvus"])
    app.include_router(chat_title_router, prefix="/chat", tags=["Chat"])

    return app