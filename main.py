from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from email_assistant import rewrite_email, EmailRequest
from ai_assistant import ask_ai, AI_Request
from anthropic import Anthropic
from pydantic import BaseModel
from typing import AsyncGenerator
import os
import uvicorn

load_dotenv()

# FastAPI app initialization
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# AI client initialization
openai_client = OpenAI()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 1000

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
    try:
        return ask_ai(request, openai_client, deepseek_client, anthropic_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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