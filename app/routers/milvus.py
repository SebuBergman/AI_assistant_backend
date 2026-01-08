
from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.db.database import MILVUS_COLLECTION_NAME, PDF_COLLECTION, QUERY_CACHE_COLLECTION, build_embeddings_schema, build_pdf_metadata_schema, build_query_cache_schema, get_milvus_collection_stats, milvus_client
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
    
@router.post("/recreate")
async def recreate_collections(confirm: bool = False):
    """
    Delete and recreate all Milvus collections with updated schemas.
    WARNING: This will delete all existing data!
    
    Args:
        Confirm: Must be set to True to proceed with recreation.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Recreation not confirmed. Set 'confirm=true' to proceed. WARNING: This will delete all data!"
        )
    
    try:
        collections_to_recreate = [
            {
                "name": MILVUS_COLLECTION_NAME,
                "schema_builder": build_embeddings_schema,
                "index_fields": [("vector", "AUTOINDEX", "COSINE")],
                "description": "RAG document embeddings"
            },
            {
                "name": PDF_COLLECTION,
                "schema_builder": build_pdf_metadata_schema,
                "index_fields": [("embedding", "AUTOINDEX", "COSINE")],
                "description": "PDF metadata"
            },
            {
                "name": QUERY_CACHE_COLLECTION,
                "schema_builder": build_query_cache_schema,
                "index_fields": [("embedding", "AUTOINDEX", "COSINE")],
                "description": "Query cache"
            }
        ]

        results = []
        
        for collection_info in collections_to_recreate:
            collection_name = collection_info["name"]
            
            try:
                # Check if collection exists
                existing_collections = milvus_client.list_collections()
                
                # Drop collection if it exists
                if collection_name in existing_collections:
                    print(f"Dropping collection: {collection_name}")
                    milvus_client.drop_collection(collection_name)
                    results.append({
                        "collection": collection_name,
                        "action": "dropped",
                        "status": "success"
                    })
                else:
                    results.append({
                        "collection": collection_name,
                        "action": "skipped_drop",
                        "status": "not_found"
                    })
                
                # Create the collection with new schema
                print(f"Creating collection: {collection_name}")
                schema = collection_info["schema_builder"]()
                
                milvus_client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    consistency_level="Strong"
                )
                
                # Add indexes
                if collection_info["index_fields"]:
                    index_params = milvus_client.prepare_index_params()
                    for field_name, index_type, metric_type in collection_info["index_fields"]:
                        index_params.add_index(
                            field_name=field_name,
                            index_type=index_type,
                            metric_type=metric_type
                        )
                    milvus_client.create_index(
                        collection_name=collection_name,
                        index_params=index_params
                    )
                
                # Load collection
                milvus_client.load_collection(collection_name)
                
                results.append({
                    "collection": collection_name,
                    "action": "created",
                    "status": "success",
                    "description": collection_info["description"]
                })
                
                print(f"✓ Collection '{collection_name}' recreated successfully")
                
            except Exception as e:
                error_msg = f"Error recreating collection '{collection_name}': {str(e)}"
                print(error_msg)
                results.append({
                    "collection": collection_name,
                    "action": "failed",
                    "status": "error",
                    "error": str(e)
                })
        
        # Check if all operations were successful
        all_success = all(
            r["status"] in ["success", "not_found"] 
            for r in results
        )
        
        return {
            "success": all_success,
            "message": "Collections recreated successfully" if all_success else "Some collections failed to recreate",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recreate collections: {str(e)}"
        )