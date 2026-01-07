from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

class EmailRequest(BaseModel):
    email: str
    tone: str

async def rewrite_email_stream(request: EmailRequest):
    email = request.email
    tone = request.tone

    print(f"Email: {email}")
    print(f"Tone: {tone}")

    prompt = f"""
    Rewrite the following email in the tone: {tone}

    Email:
    {email}

    Rules:
    - Match the requested tone.
    - Use an appropriate email structure.
    """

    print(f"Prompt: {prompt}")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True
    )

    messages = [("user", prompt)]

    async def stream_generator():
        try:
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    # Send as JSON object
                    import json
                    yield f"data: {json.dumps({'content': chunk.content})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            print(f"Stream error: {str(e)}")
            import traceback
            traceback.print_exc()
            import json
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")