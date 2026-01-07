from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

router = APIRouter()

class TitleRequest(BaseModel):
    message: str
    chat_id: str

class TitleResponse(BaseModel):
    title: str
    chat_id: str

@router.post("/title", response_model=TitleResponse)
async def generate_chat_title(request: TitleRequest):
    """
    Generate a smart title for a chat based on the first message
    Uses GPT-4o-mini for cost efficiency
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Truncate message to save tokens
        message_preview = request.message[:200]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast!
            max_tokens=50,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": f'Generate a brief, descriptive title (maximum 6 words) for a chat that starts with this message: "{message_preview}"\n\nRespond with ONLY the title, no quotes or extra punctuation.'
            }]
        )
        
        # Extract title from response
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # Validate title
        if not title or len(title) > 100:
            # Fallback to truncated message
            title = request.message[:60].strip()
            if len(request.message) > 60:
                title += "..."
        
        return TitleResponse(title=title, chat_id=request.chat_id)
        
    except Exception as e:
        print(f"Error generating title: {e}")
        # Return fallback title on error
        fallback = request.message[:60].strip()
        if len(request.message) > 60:
            fallback += "..."
        return TitleResponse(title=fallback, chat_id=request.chat_id)