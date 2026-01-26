import json
import tiktoken

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
            context = ""
            references = []
            
            # If RAG is enabled, fetch context and augment prompt
            if request.ragEnabled:
                rag_result = get_rag_context(
                    question=request.prompt,
                    file_name=request.file_name or "",
                    keyword=request.keyword or "",
                    cached=request.cached or False,
                    alpha=request.alpha or 0.7
                )

                if rag_result and rag_result.get("context_text"):
                    # Structured references (NO string parsing)
                    references = []

                    for doc in rag_result["documents"]:
                        references.append({
                            "file_name": doc.get("file_name"),
                            "content": (doc.get("text") or doc.get("content", ""))[:500],
                            "score": (
                                doc.get("hybrid_score")
                                or doc.get("score")
                            )
                        })

                    # Set context for sending to AI_request
                    context=rag_result["context_text"]

                    # Send metadata once at start of stream
                    yield f"data: {json.dumps({
                        'metadata': {
                            'rag_enabled': True,
                            'search_method': rag_result['search_method'],
                            'references': references
                        }
                    })}\n\n"
                    
            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(prompt)) # Count input tokens
            output_tokens = 0

            # Create a modified request with the augmented prompt
            ai_request = AI_Request(
                rag_enabled=request.ragEnabled,
                question=prompt,
                context=context,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            print(f"AI_request: {ai_request}")
            
            # Get the streaming generator from ask_ai
            stream = ask_ai(
                ai_request,
                openai_client,
                deepseek_client,
                anthropic_client
            )

            # Stream the content with SSE formatting
            for chunk in stream:
                if isinstance(chunk, str) and not chunk.startswith('{'):
                    # Count tokens in each chunk
                    output_tokens += len(encoding.encode(chunk))

                # Stream chunks
                if isinstance(chunk, str):
                    if chunk.startswith('{'):
                        yield f"data: {chunk}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': str(chunk)})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({
                'done': True,
                'tokens': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                }
            })}\n\n"

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

def build_normal_prompt(user_question: str) -> str:
    return f"""You are a helpful assistant.
        Answer the user's question clearly and directly.

        Rules:
        - Do not invent facts. If you don't know, say so. If something needs verification, state what you would need to confirm.
        - Use Markdown formatting in your responses (headings, lists, code blocks, etc.) so the output renders well in react-markdown.
        - If the question is ambiguous or missing key details, ask 1-3 brief clarifying questions.
        - Use plain language unless the user is clearly asking for technical depth.

        User question: {user_question}

        Answer:
    """.strip()


def build_rag_prompt(user_question: str, context: str) -> str:
    return f"""
        You are a retrieval-augmented assistant. Answer the user's question using ONLY the information in the provided Context.

        Rules:
        - Use ONLY the Context to answer. Do not use prior knowledge or make assumptions.
        - If the Context does not contain enough information to answer, say: "I don't know based on the provided context."
        - Be concise, factual, and specific. Prefer short direct statements.
        - Include citations, but keep them minimal (e.g., 1-3 per answer) and attach them only to the specific claims they support.
        - Do not quote or paste large portions of the retrieved materials; summarize in your own words.
        - Do not mention or describe the retrieved materials, “context,” “documents,” or the retrieval process.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
    """.strip()

@router.get("/models")
async def list_models():
    """Get list of available AI models"""
    from app.assistants.ai_assistant import MODEL_FUNCTIONS
    return {"models": list(MODEL_FUNCTIONS.keys())}

@router.get("/tools/{model_name}")
async def check_tool_support(model_name: str):
    """Check if a specific model supports tool calling"""
    return {
        "model": model_name,
        "supports_tools": is_tool_supported(model_name)
    }