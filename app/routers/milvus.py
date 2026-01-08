
from fastapi import APIRouter, HTTPException

from app.db.database import get_milvus_collection_stats

router = APIRouter()

@router.get("/stats", tags=["Milvus"])
def milvus_stats():
    """Get Milvus/Zilliz collection statistics"""
    try:
        stats = get_milvus_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema", tags=["Milvus"])
def get_collection_schema():
    """Get the current collection schema to verify it's correct"""
    try:
        from db.database import milvus_client, MILVUS_COLLECTION_NAME
        
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