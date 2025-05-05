from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from email_assistant import rewrite_email, EmailRequest
from ai_assistant import ask_ai, AI_Request

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# OpenAI client initialization
openai_client = OpenAI()

# CORS configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return ask_ai(request, openai_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))