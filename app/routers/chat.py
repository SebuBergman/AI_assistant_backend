import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.dependencies import openai_client, deepseek_client, anthropic_client
from app.tools.tools import is_tool_supported
from app.assistants.ai_assistant import ask_ai, AI_Request
from app.utils.rag_helpers import get_rag_context

router = APIRouter()

class ExtendedAI_Request(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 20240
    ragEnabled: bool = False
    file_name: str = ""
    keyword: str = ""
    cached: bool = False
    alpha: float = 0.7

@router.post("/generate")
async def ask_ai_endpoint(request: ExtendedAI_Request):
    """Unified streaming generator with optional RAG support"""
    
    async def generate():
        try:
            # Prepare the prompt
            prompt = request.prompt
            references = []
            
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
                    # Parse context into structured references for frontend
                    # Context format: "Document 1 (Score: 0.85)\nFile: example.pdf\nContent here...\n"
                    context_lines = context.split('\n\n')
                    for doc_block in context_lines:
                        if doc_block.strip():
                            lines = doc_block.split('\n')
                            if len(lines) >= 3:
                                # Extract document info
                                doc_header = lines[0]  # "Document 1 (Score: 0.85)" or "Document 1 (Hybrid: 0.85)"
                                file_line = lines[1]    # "File: example.pdf"
                                content = '\n'.join(lines[2:])  # Rest is content
                                
                                # Parse file name
                                file_name = file_line.replace('File: ', '').strip()
                                
                                # Parse score
                                score = None
                                if 'Score:' in doc_header:
                                    score = doc_header.split('Score:')[1].strip().rstrip(')')
                                elif 'Hybrid:' in doc_header:
                                    score = doc_header.split('Hybrid:')[1].strip().rstrip(')')
                                
                                references.append({
                                    'file_name': file_name,
                                    'content': content[:500],  # Truncate for preview
                                    'score': score
                                })
                    
                    # Augment the prompt with RAG context
                    prompt = f"""Use the following context to help answer the question.
                    If the context is relevant, use it. If not, answer based on your knowledge.

                    Context:
                    {context}

                    Question: {request.prompt}

                    Answer:"""
                    
                    # Send metadata with structured references
                    yield f"data: {json.dumps({
                        'metadata': {
                            'rag_enabled': True, 
                            'search_method': search_method,
                            'references': references
                        }
                    })}\n\n"
            
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

@router.get("/models")
async def list_models():
    """Get list of available AI models"""
    from assistants.ai_assistant import MODEL_FUNCTIONS
    return {"models": list(MODEL_FUNCTIONS.keys())}

@router.get("/tools/{model_name}")
async def check_tool_support(model_name: str):
    """Check if a specific model supports tool calling"""
    return {
        "model": model_name,
        "supports_tools": is_tool_supported(model_name)
    }