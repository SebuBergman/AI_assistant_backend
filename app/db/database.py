import os
import time
import numpy as np
import ssl
import asyncio
import sys
import asyncpg
import redis.asyncio as aioredis

from typing import Optional
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from datetime import datetime
from contextlib import asynccontextmanager

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

USE_BACKEND_POOLS = False

# Global connection pools
_db_pool: Optional[asyncpg.Pool] = None
_redis_client: Optional[aioredis.Redis] = None

# Cache configuration
CACHE_KEYS = {
    "user_chats": lambda user_id: f"chats:user:{user_id}",
    "chat_meta": lambda chat_id: f"chat:meta:{chat_id}",
    "chat_messages": lambda chat_id: f"chat:messages:{chat_id}",
}

CACHE_TTL = {
    "chats": 300,      # 5 minutes
    "messages": 300,   # 5 minutes
}

# Zilliz Cloud / Milvus config
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")

# Milvus collection names
MILVUS_COLLECTION_NAME = os.getenv("EMBEDDINGS_COLLECTION_NAME", "embeddings")
PDF_COLLECTION = os.getenv("PDFS_COLLECTION_NAME", "pdf_metadata")
QUERY_CACHE_COLLECTION = os.getenv("QUERY_CACHE_COLLECTION_NAME", "query_cache")

# Embedding dimension
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Initialize Milvus client for Zilliz Cloud
milvus_client = MilvusClient(
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_TOKEN
)

# Test connection
try:
    collections = milvus_client.list_collections()
    print(f"✓ Zilliz Cloud connection successful. Collections: {collections}")
except Exception as e:
    print(f"✗ Zilliz Cloud connection failed: {e}")
    raise

# =========================
# Milvus Collection Helpers
# =========================
def ensure_collection_exists(name: str, schema_builder, index_fields: list = []):
    """Create a collection only if it doesn't exist."""
    try:
        existing = milvus_client.list_collections()
        if name in existing:
            print(f"✓ Collection already exists: {name}")
            return

        print(f"Creating new collection: {name}")

        # Create schema using the provided builder
        schema = schema_builder()

        # Create the collection
        milvus_client.create_collection(
            collection_name=name,
            schema=schema,
            consistency_level="Strong"
        )

        # Add indexes
        if index_fields:
            index_params = milvus_client.prepare_index_params()
            for field_name, index_type, metric_type in index_fields:
                index_params.add_index(
                    field_name=field_name,
                    index_type=index_type,
                    metric_type=metric_type
                )
            milvus_client.create_index(collection_name=name, index_params=index_params)

        # Load collection
        milvus_client.load_collection(name)
        print(f"✓ Collection '{name}' created and loaded successfully")
    except Exception as e:
        print(f"Error creating collection '{name}': {e}")
        raise

# =========================
# Schema Builders
# =========================

def build_embeddings_schema():
    schema = milvus_client.create_schema(
        auto_id=True, enable_dynamic_field=True, description="RAG document embeddings"
    )
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)

    # content + embedding
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    # document-level metadata
    schema.add_field("file_name", DataType.VARCHAR, max_length=512)
    schema.add_field("file_id", DataType.VARCHAR, max_length=128)
    schema.add_field("file_size", DataType.INT64)
    schema.add_field("upload_date", DataType.VARCHAR, max_length=32)
    
    # Chunk-level metadata (used by fetch_documents)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("page", DataType.INT64)

    # Chunk tokens
    schema.add_field("chunk_tokens", DataType.INT64)
    return schema

def build_pdf_metadata_schema():
    schema = milvus_client.create_schema(
        auto_id=True, description="PDF metadata"
    )
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("file_name", DataType.VARCHAR, max_length=512)
    schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("upload_date", DataType.VARCHAR, max_length=32)
    schema.add_field("file_size", DataType.INT64)
    schema.add_field("chunks", DataType.INT64)
    schema.add_field("file_id", DataType.VARCHAR, max_length=128)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)
    return schema

def build_query_cache_schema():
    schema = milvus_client.create_schema(
        auto_id=True, description="Query cache"
    )
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("query", DataType.VARCHAR, max_length=2048)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("answer", DataType.VARCHAR, max_length=65535)
    schema.add_field("context", DataType.VARCHAR, max_length=65535)
    schema.add_field("timestamp", DataType.INT64)
    return schema

# =========================
# Ensure All Collections Exist
# =========================

# Embeddings
ensure_collection_exists(
    MILVUS_COLLECTION_NAME,
    build_embeddings_schema,
    index_fields=[("vector", "AUTOINDEX", "COSINE")]
)

# PDF Metadata
ensure_collection_exists(
    PDF_COLLECTION,
    build_pdf_metadata_schema,
    index_fields=[("embedding", "AUTOINDEX", "COSINE")]
)

# Query Cache
ensure_collection_exists(
    QUERY_CACHE_COLLECTION,
    build_query_cache_schema,
    index_fields=[("embedding", "AUTOINDEX", "COSINE")]
)

# PDF Metadata operations
def insert_pdf_metadata(file_name, file_path, file_size, upload_date, file_id, chunks, embedding=None):
    """Store PDF metadata into Milvus."""

    if embedding is None:
        embedding = [0.0] * 384  # Dummy embedding if no vector available

    milvus_client.insert(
        collection_name=PDF_COLLECTION,
        data=[{
            "file_name": file_name,
            "file_path": file_path,
            "upload_date": upload_date,
            "file_size": file_size,
            "chunks": chunks,
            "file_id": file_id,
            "embedding": embedding
        }]
    )
    print(f"✓ Saved PDF metadata for {file_name}")

def get_pdf_metadata():
    """Return all PDF metadata records with readable timestamps."""
    results = milvus_client.query(
        collection_name=PDF_COLLECTION,
        filter="pk >= 0",
        output_fields=["file_name", "file_path", "upload_date", "file_size", "chunks", "file_id"]
    )

    # Convert timestamp to readable datetime
    for record in results:
        if "upload_date" in record and isinstance(record["upload_date"], datetime):
            record["upload_date"] = record["upload_date"].strftime("%Y-%m-%d %H:%M:%S")

    return results
    
# Query cache operations 
def store_query_result(query_text, embedding, answer, context):
    """Insert query cache entry into Milvus."""
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    milvus_client.insert(
        collection_name=QUERY_CACHE_COLLECTION,
        data=[{
            "query": query_text,
            "embedding": embedding,
            "answer": answer,
            "context": context,
            "timestamp": int(time.time())
        }]
    )
    print("✓ Cached query")
    
def find_similar_cached_query(query_embedding, threshold=0.85):
    """Vector search inside Milvus query_cache collection."""
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    results = milvus_client.search(
        collection_name=QUERY_CACHE_COLLECTION,
        data=[query_embedding],
        anns_field="embedding",
        limit=3,
        output_fields=["query", "answer", "context", "timestamp"],
    )

    top = results[0]
    if not top:
        return None

    hit = top[0]
    if hit["distance"] >= threshold:  # cosine similarity
        return {
            "query": hit["entity"]["query"],
            "answer": hit["entity"]["answer"],
            "context": hit["entity"]["context"],
            "timestamp": hit["entity"]["timestamp"]
        }

    return None

# Delete operations
# Single document deletions
def delete_document_embeddings(file_name: str):
    """Delete all embeddings for a specific document from Milvus."""
    try:
        if MILVUS_COLLECTION_NAME not in milvus_client.list_collections():
            print(f"Collection '{MILVUS_COLLECTION_NAME}' does not exist.")
            return 0
        
        # Delete entities where source matches the file_name
        result = milvus_client.delete(
            collection_name=MILVUS_COLLECTION_NAME,
            filter=f'source == "{file_name}"'
        )
        
        deleted_count = result.get('delete_count', 0) if isinstance(result, dict) else 0
        print(f"✓ Deleted {deleted_count} embeddings for document '{file_name}'")
        return deleted_count
    except Exception as e:
        print(f"Error deleting embeddings for '{file_name}': {str(e)}")
        raise

def delete_pdf_metadata(file_name: str):
    """Delete PDF metadata for a specific document."""
    try:
        if PDF_COLLECTION not in milvus_client.list_collections():
            print(f"PDF metadata collection does not exist.")
            return False
        
        result = milvus_client.delete(
            collection_name=PDF_COLLECTION,
            filter=f'file_name == "{file_name}"'
        )
        
        print(f"✓ Deleted PDF metadata for '{file_name}'")
        return True
    except Exception as e:
        print(f"Error deleting PDF metadata for '{file_name}': {str(e)}")
        raise

def clear_document_cache(file_name: str):
    """Clear cache entries for a specific document."""
    try:
        if QUERY_CACHE_COLLECTION not in milvus_client.list_collections():
            print(f"Query cache collection does not exist.")
            return
        
        # Delete cache entries where the response references this document
        # This assumes your cache stores which documents were used in responses
        result = milvus_client.delete(
            collection_name=QUERY_CACHE_COLLECTION,
            filter=f'source == "{file_name}"'
        )
        
        print(f"✓ Cleared cache entries for document '{file_name}'")
    except Exception as e:
        print(f"Error clearing cache for '{file_name}': {e}")
        # Non-critical error, so we can continue

# Bulk deletions
def clear_all_pdfs():
    """Delete all PDF metadata."""
    if PDF_COLLECTION in milvus_client.list_collections():
        milvus_client.drop_collection(PDF_COLLECTION)
        build_pdf_metadata_schema()
        print("✓ Cleared all PDF metadata")

def clear_cache_entries():
    """Reset entire query cache."""
    milvus_client.drop_collection(QUERY_CACHE_COLLECTION)
    build_query_cache_schema()
    print("✓ Cleared query cache")

# Clear data functions
def clear_all_embeddings():
    """Clear all embeddings from Milvus."""
    try:
        # Check if collection exists before dropping
        if MILVUS_COLLECTION_NAME in milvus_client.list_collections():
            milvus_client.drop_collection(MILVUS_COLLECTION_NAME)
            print(f"✓ Dropped Milvus collection '{MILVUS_COLLECTION_NAME}' successfully.")
        else:
            print(f"Collection '{MILVUS_COLLECTION_NAME}' does not exist.")
        
        # Recreate the collection
        build_embeddings_schema()
        return True
    except Exception as e:
        print(f"Error clearing all embeddings: {str(e)}")
        raise

# Milvus statistics
def get_milvus_collection_stats():
    """Get statistics about the Milvus collection"""
    try:
        stats = milvus_client.get_collection_stats(MILVUS_COLLECTION_NAME)
        return {
            "collection_name": MILVUS_COLLECTION_NAME,
            "stats": stats
        }
    except Exception as e:
        print(f"Error getting Milvus stats: {str(e)}")
        return {"error": str(e)}
    
def get_cache_stats():
    stats = milvus_client.get_collection_stats(QUERY_CACHE_COLLECTION)
    return stats


# Database connection pool management for Supabase/PostgreSQL
async def init_db_pool():
    """
    Initialize the PostgreSQL connection pool with SSL verification using a
    custom CA (Option 2). Suitable for production with Supabase.
    """
    if not USE_BACKEND_POOLS:
        print("⚠️ Skipping backend DB pool initialization")
        return None

    global _db_pool

    if _db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not set")

        # Optional: path to Supabase root CA certificate
        # Download from https://supabase.com/docs/reference/cli or dashboard
        supabase_ca_path = os.getenv("SUPABASE_CA_PATH", "supabase-ca.pem")

        ssl_context = ssl.create_default_context(cafile=supabase_ca_path)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        _db_pool = await asyncpg.create_pool(
            database_url,
            ssl=ssl_context,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )

        print("✅ Database pool initialized with verified SSL")

    return _db_pool


async def close_db_pool():
    """Close the database connection pool"""
    global _db_pool

    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None
        print("Database pool closed")


async def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool"""
    if _db_pool is None:
        await init_db_pool()
    return _db_pool


#Redis connection management
async def init_redis():
    if not USE_BACKEND_POOLS:
        print("⚠️ Skipping backend DB pool initialization")
        return None
    
    global _redis_client

    if _redis_client is None:
        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
        password = os.getenv("REDIS_PASSWORD")

        if not host or not port:
            raise ValueError("REDIS_HOST and REDIS_PORT must be set")

        _redis_client = aioredis.Redis(
            host=host,
            port=int(port),
            password=password,
            username="default",
            decode_responses=True,
        )

        await _redis_client.ping()
        print("Redis client initialized")

async def close_redis():
    """Close Redis connection"""
    global _redis_client
    
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        print("Redis client closed")

async def get_redis_client() -> aioredis.Redis:
    """Get Redis client"""
    if _redis_client is None:
        await init_redis()
    return _redis_client

# Context manager for FastAPI lifespan
@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager"""
    # Startup
    await init_db_pool()
    await init_redis()
    print("Application started, connections initialized")
    
    yield
    
    # Shutdown
    await close_db_pool()
    await close_redis()
    print("Application shutdown, connections closed")