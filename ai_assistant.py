from openai import OpenAI
from pydantic import BaseModel
from typing import Dict

class AI_Request(BaseModel):
    question: str
    model: str
    temperature: float = 0.7  # Default temperature

# DeepSeek model functions remain the same as before
def deepseek_chat(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI):
    system_prompt = """
    I want you to act as a AI assistant that answers the user's prompt in a friendly and helpful manner.
    
    Rules:
    - Be as helpful as possible
    - Create profound answers
    - Maintain conversational tone
    """
    try:
        deepseek_response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            temperature=request.temperature
        )
        return {"answer": deepseek_response.choices[0].message.content}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}

def deepseek_reasoner(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI):
    system_prompt = """
    I want you to act as a reasoning-focused AI assistant.
    
    Rules:
    - Focus on step-by-step reasoning
    - Provide thorough explanations
    - Be precise
    - Include logical frameworks
    """
    try:
        openai_response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            temperature=request.temperature
        )
        return {"answer": openai_response.choices[0].message.content}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}
    
def claude_code_assistant(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI, anthropic_client):
    system_prompt = """
    I want you to act as a programming-focused AI assistant.
    
    Rules:
    - Focus on code completion and debugging
    - Provide clear code examples
    - Be precise
    - Be concise
    """
    try:
        claude_response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ]
        )
        return {"answer": claude_response.content[0].text}
    except Exception as e:
        print(f"Claude API error: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}

def gpt_models(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI):
    """Handle all GPT model requests with dynamic model selection"""
    print(f"Received request for model: {request.model}")
    # Validate the model name
    valid_gpt_models = {
        "gpt-4.1": "GPT-4.1 (Comprehensive answers with professional tone)",
        "gpt-4.1-mini": "GPT-4.1 Mini (Concise but informative responses)",
        "gpt-4-nano": "GPT-4 Nano (Very short 1-2 sentence answers)",
        "gpt-4o": "GPT-4o (Sophisticated, nuanced responses)",
        "gpt-4o-mini": "GPT-4o Mini (Balanced depth and efficiency)"
    }
    
    if request.model not in valid_gpt_models:
        return {"answer": f"Error: Unsupported GPT model '{request.model}'"}

    # System prompt tailored for GPT models
    system_prompt = f"""
    You are {valid_gpt_models[request.model]}. Provide the best possible answer to the user's question.
    
    Guidelines:
    - Respond to the user's query appropriately for your model type
    - Maintain appropriate response length
    - Adapt to the user's requested temperature: {request.temperature}
    - Be helpful and accurate
    """
    
    try:
        openai_response = openai_client.chat.completions.create(
            model=request.model,  # Dynamic model selection
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": request.question}
            ],
            temperature=request.temperature  # Dynamic temperature
        )
        return {"answer": openai_response.choices[0].message.content}
    except Exception as e:
        print(f"OpenAI API error with model {request.model}: {str(e)}")
        return {"answer": "I encountered an error while processing your question."}

MODEL_FUNCTIONS = {
    "deepseek-chat": deepseek_chat,
    "deepseek-reasoner": deepseek_reasoner,
    "gpt-4.1": gpt_models,
    "gpt-4.1-mini": gpt_models,
    "gpt-4-nano": gpt_models,
    "gpt-4o": gpt_models,
    "gpt-4o-mini": gpt_models,
    "claude-code-assistant": claude_code_assistant
}

def ask_ai(request: AI_Request, openai_client: OpenAI, deepseek_client: OpenAI, anthropic_client):
    """Route to the appropriate model function"""
    try:
        if request.model not in MODEL_FUNCTIONS:
            print(f"Unsupported model: {request.model}")
            raise ValueError(f"Unsupported model: {request.model}")
        
        print(f"Processing request with model: {request.model}")
        # Ensure temperature is within valid range (0-2)
        request.temperature = max(0.0, min(2.0, request.temperature))
        
        return MODEL_FUNCTIONS[request.model](request, openai_client, deepseek_client, anthropic_client)
    except Exception as e:
        print(f"Error in AI processing: {str(e)}")
        return {"answer": f"System error: {str(e)}"}