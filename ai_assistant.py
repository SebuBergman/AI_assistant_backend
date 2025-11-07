from openai import OpenAI
from pydantic import BaseModel
from typing import Dict
from typing import AsyncGenerator
import json

from tools import ALL_TOOLS, TOOL_FUNCTIONS

class AI_Request(BaseModel):
    question: str
    model: str
    temperature: float = 0.7  # Default temperature
    max_tokens: int = 1024  # Default max tokens

# DeepSeek model functions remain the same as before
def deepseek_chat_stream(request: AI_Request, **kwargs) -> AsyncGenerator:
    client = kwargs.get('deepseek_client')
    system_prompt = """
    I want you to act as a AI assistant that answers the user's prompt in a friendly and helpful manner.
    
    Rules:
    - Be as helpful as possible
    - Create profound answers
    - Maintain conversational tone
    """
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            temperature=request.temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"DeepSeek chat error: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}

def deepseek_reasoner_stream(request: AI_Request, **kwargs) -> AsyncGenerator:
    client = kwargs.get('deepseek_client')
    system_prompt = """
    I want you to act as a reasoning-focused AI assistant.
    
    Rules:
    - Focus on step-by-step reasoning
    - Provide thorough explanations
    - Be precise
    - Include logical frameworks
    """
    try:
        stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            stream=True,
            temperature=request.temperature
        )
        
        for chunk in stream:
            # Handle reasoning content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                yield json.dumps({
                    "type": "reasoning",
                    "content": chunk.choices[0].delta.reasoning_content
                }) + "\n"

            # Handle regular content
            if chunk.choices[0].delta.content:
                yield json.dumps({
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }) + "\n"
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}
    
def claude_models_stream(request: AI_Request, **kwargs) -> AsyncGenerator:
    """Handle Claude Code Assistant requests with dynamic model selection"""
    client = kwargs.get('anthropic_client')
    print(f"Received request for model: {request.model}")
    # Validate the model name
    valid_claude_models = {
    "claude-4.5-sonnet": "Claude Sonnet 4.5 (Next-gen Sonnet tier: high throughput, production workloads with strong coding & reasoning) ",
    "claude-4.1-opus": "Claude Opus 4.1 (Top-tier flagship: complex reasoning, long context, full-agent capability and highest accuracy) ",
    "claude-3.5-haiku": "Claude 3.5 Haiku (Fast and cost-efficient; good for everyday queries, moderation & translation)",
    "claude-3.5-sonnet": "Claude 3.5 Sonnet (Balanced performance and cost; capable of deeper reasoning & data tasks)",
    "claude-3.7-sonnet": "Claude 3.7 Sonnet (Hybrid reasoning mode: choose speed vs depth; advanced coding and agent workflows) ",
    }

    if request.model not in valid_claude_models:
        return {"answer": f"Error: Unsupported GPT model '{request.model}'"}

    system_prompt = f"""
    You are {valid_claude_models[request.model]}.
    I want you to act as a programming-focused AI assistant.
    
    Rules:
    - Focus on code completion and debugging
    - Provide clear code examples
    - Be precise
    - Be concise
    """
    try:
        with client.messages.stream(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            temperature=request.temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        print(f"OpenAI API error with model {request.model}: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}

def gpt_models_stream(request: AI_Request, **kwargs) -> AsyncGenerator:
    """Handle all GPT model requests with dynamic model selection"""
    client = kwargs.get('openai_client')
    print(f"Received request for model: {request.model}")
    # Validate the model name
    valid_gpt_models = {
        "gpt-5": "GPT-5 (Advanced reasoning and creativity with expert-level responses)",
        "gpt-5-mini": "GPT-5 Mini (Fast, efficient, and concise with solid reasoning)",
        "gpt-5-nano": "GPT-5 Nano (Ultra-light, delivers brief 1–2 sentence answers)",
        "gpt-4.1": "GPT-4.1 (Comprehensive answers with professional tone)",
        "gpt-4.1-mini": "GPT-4.1 Mini (Concise but informative responses)",
        "gpt-4.1-nano": "GPT-4.1 Nano (Very short 1-2 sentence answers)",
        "gpt-4o": "GPT-4o (Sophisticated, nuanced responses)",
    }
    
    if request.model not in valid_gpt_models:
        return {"answer": f"Error: Unsupported GPT model '{request.model}'"}

    # System prompt tailored for GPT models
    system_prompt = f"""
    You are {valid_gpt_models[request.model]}.
    Provide the best possible answer to the user's question.
    
    Guidelines:
    - Respond to the user's query appropriately for your model type
    - Maintain appropriate response length
    - Adapt to the user's requested temperature: {request.temperature}
    - Be helpful and accurate
    """

    supports_tools = not any(x in request.model for x in ["nano", "mini"])
    
    try:
        kwargs = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            "temperature": request.temperature,
            "stream": True
        }

        if supports_tools:
            kwargs["tools"] = ALL_TOOLS

        stream = client.chat.completions.create(**kwargs)

        # ✅ Stream only if stream is iterable
        if not stream:
            yield {"answer": "No response stream received from the API."}
            return

        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"OpenAI API error with model {request.model}: {str(e)}")
        yield {"answer": "I encountered an error while processing your question."}

MODEL_FUNCTIONS = {
    "deepseek-chat": deepseek_chat_stream,
    "deepseek-reasoner": deepseek_reasoner_stream,
    "gpt-5": gpt_models_stream,
    "gpt-5-mini": gpt_models_stream,
    "gpt-5-nano": gpt_models_stream,
    "gpt-4.1": gpt_models_stream,
    "gpt-4.1-mini": gpt_models_stream,
    "gpt-4.1-nano": gpt_models_stream,
    "gpt-4o": gpt_models_stream,
    "claude-3.5-haiku": claude_models_stream,
    "claude-3.5-sonnet": claude_models_stream,
    "claude-3.7-sonnet": claude_models_stream,
    "claude-4.5-sonnet": claude_models_stream,
    "claude-4.1-opus": claude_models_stream,
}

def ask_ai(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI, anthropic_client):
    """Route to the appropriate model function (streaming version)"""
    try:
        if request.model not in MODEL_FUNCTIONS:
            print(f"Unsupported model: {request.model}")
            raise ValueError(f"Unsupported model: {request.model}")
        
        print(f"Processing streaming request with model: {request.model}")
        request.temperature = max(0.0, min(2.0, request.temperature))
        
        # Return the generator directly for streaming
        return MODEL_FUNCTIONS[request.model](
            request, 
            openai_client=openai_client, 
            deepseek_client=deepseek_client, 
            anthropic_client=anthropic_client
        )
    except Exception as e:
        print(f"Error in AI streaming: {str(e)}")
        raise