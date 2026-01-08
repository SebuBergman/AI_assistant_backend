from fastapi import APIRouter, HTTPException

from app.assistants.email_assistant import EmailRequest, rewrite_email_stream

router = APIRouter()

@router.post("/email_assistant", tags=["Email"])
async def email_assistant_endpoint(request: EmailRequest):
    """Rewrite emails in different tones"""
    try:
        return await rewrite_email_stream(request)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))