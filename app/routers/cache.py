import datetime
from fastapi import APIRouter, HTTPException

from app.db.database import QUERY_CACHE_COLLECTION, clear_cache_entries, get_cache_stats
from app.config import milvus_client

router = APIRouter()

@router.get("/stats", tags=["Cache"])
def cache_stats_endpoint():
    """Get cache statistics"""
    try:
        stats = get_cache_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear_old", tags=["Cache"])
def clear_old_cache_entries_endpoint(days: int = 30):
    """Clear cache entries older than specified days (default: 30)"""
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

@router.post("/clear_all", tags=["Cache"])
def clear_whole_cache():
    """Clear ALL cache entries"""
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

@router.get("/entries", tags=["Cache"])
def list_cache_entries(limit: int = 10):
    """List recent cached queries from Milvus"""
    try:
        results = milvus_client.query(
            collection_name=QUERY_CACHE_COLLECTION,
            filter="pk >= 0",
            output_fields=["query", "answer", "timestamp"],
            limit=limit
        )

        results = sorted(results, key=lambda x: x["timestamp"], reverse=True)

        for r in results:
            r["timestamp"] = datetime.fromtimestamp(r["timestamp"]).isoformat()

        return {"entries": results, "count": len(results)}

    except Exception as e:
        print(f"Error listing cache entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))