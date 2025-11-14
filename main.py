from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from email_assistant import rewrite_email, EmailRequest
from ai_assistant import ask_ai, AI_Request
from anthropic import Anthropic
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import os
import uvicorn
import json

from tools import is_tool_supported

load_dotenv()

# FastAPI app initialization
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# AI client initialization
openai_client = OpenAI()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 1024

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Assistant API"}

@app.post("/email_assistant")
async def email_assistant_endpoint(request: EmailRequest):
    try:
        return rewrite_email(request, openai_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_ai")
async def ask_ai_endpoint(request: AI_Request):
    """Unified streaming generator that wraps ai_assistant functions"""
    
    async def generate():
        try:
            # Get the streaming generator from ask_ai
            stream = ask_ai(
                request,
                openai_client,
                deepseek_client,
                anthropic_client
            )

            # Stream the content with SSE formatting
            for chunk in stream:
                # Handle both string chunks and JSON chunks (for reasoner)
                if isinstance(chunk, str):
                    if chunk.startswith('{'):
                        # Already JSON formatted (from deepseek-reasoner)
                        yield f"data: {chunk}\n\n"
                    else:
                        # Plain text content
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
    from tools import is_tool_supported
    return {
        "model": model_name,
        "supports_tools": is_tool_supported(model_name)
    }

def main():
    """Main function to run the FastAPI application with uvicorn server."""
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
    
if __name__ == "__main__":
    main()