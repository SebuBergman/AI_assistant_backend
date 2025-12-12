from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from fastapi import StreamingResponse

import SSEStreamer

class EmailRequest(BaseModel):
    email: str
    tone: str

def rewrite_email_stream(request: EmailRequest):
    email = request.email
    tone = request.tone

    prompt = f"""
    Rewrite the following email in the tone: {tone}

    Email:
    {email}

    Rules:
    - Match the requested tone.
    - Use an appropriate email structure.
    """

    # Create a streaming callback
    callback = SSEStreamer()

    # Initialize LangChain ChatOpenAI with streaming enabled
    llm = ChatOpenAI(
        model="gpt-5-mini",
        streaming=True,
        callbacks=[callback]
    )

    # LangChain expects an array of messages
    messages = [
        ("user", prompt),
    ]

    def stream_generator():
        try:
            # Trigger the streaming run
            llm.invoke(messages)

            # After invoke starts, stream tokens as they appear
            while True:
                tokens = callback.get_tokens()
                if tokens:
                    yield f"data: {tokens}\n\n"
                else:
                    break

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: STREAM_ERROR: {str(e)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
