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
        You are an email rewriting assistant.

        Task:
        Rewrite the email below in this tone: {tone}

        Input email:
        ---
        {email}
        ---

        Guidelines:
        - Preserve the original meaning and key details (facts, dates, names, requests).
        - Match the requested tone consistently.
        - Improve clarity and flow; fix grammar and awkward phrasing.
        - Keep it concise unless the original is long or detailed.
        - Use a standard email structure:
        - Subject (if one is implied or helpful)
        - Greeting
        - Body
        - Closing/sign-off
        - Do not add new information, promises, or commitments that are not in the original.

        Output:
        Return only the rewritten email text (no commentary).
    """.strip()

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