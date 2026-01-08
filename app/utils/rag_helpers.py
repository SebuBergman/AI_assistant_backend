from app.db.database import find_similar_cached_query
from app.db.vectorstore_manager import get_vectorstore, embeddings
from app.rag.rag_search import hybrid_search, keyword_search, vector_search

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
            # ✅ Use .get() with fallback to handle both 'text' and 'content' fields
            content = result.get('text') or result.get('content', '')
            
            if search_method == "hybrid":
                context_lines.append(
                    f"Document {i+1} (Hybrid: {result['hybrid_score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{content}\n"
                )
            else:
                context_lines.append(
                    f"Document {i+1} (Score: {result['score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{content}\n"
                )
        
        context = "\n".join(context_lines)
        return context, search_method
    
    except Exception as e:
        print(f"Error getting RAG context: {e}")
        return None, f"Error: {str(e)}"