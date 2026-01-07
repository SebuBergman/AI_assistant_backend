from openai import OpenAI
from pydantic import BaseModel
from typing import AsyncGenerator
import json
import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from data import valid_gpt_models, valid_claude_models
from tools import LANGCHAIN_TOOLS, is_tool_supported

class AI_Request(BaseModel):
    question: str
    model: str
    temperature: float = 1
    max_tokens: int = 10240

# DeepSeek model functions remain the same as before
def deepseek_chat_stream(request: AI_Request, **kwargs) -> AsyncGenerator: # type: ignore
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

def deepseek_reasoner_stream(request: AI_Request, **kwargs) -> AsyncGenerator: # type: ignore
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
    
def claude_models_stream(request: AI_Request, **kwargs) -> AsyncGenerator: # type: ignore
    """Handle Claude models with LangCgain tool calling support and streaming"""

    print(f"Received request for model: {request.model}")

    # Validate the model name
    if request.model not in valid_claude_models:
        yield {"answer": f"Error: Unsupported GPT model '{request.model}'"}
        return

    try:
        # Initialize LangChain ChatAnthropic
        llm = ChatAnthropic(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Check if this model supports tools
        supports_tools = is_tool_supported(request.model)
        
        # Bind tools to the model if supported
        if supports_tools:
            llm_with_tools = llm.bind_tools(LANGCHAIN_TOOLS)
            print(f"[Claude] Tools enabled for {request.model}")
        else:
            llm_with_tools = llm
            print(f"[Claude] No tool support for {request.model}")
        
        # Initialize conversation messages
        messages = [HumanMessage(content=request.question)]
        
        # Agent loop for handling tool calls
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"[Claude] Iteration {iteration}")
            
            # Invoke the model
            response = llm_with_tools.invoke(messages)
            
            # Stream the text content
            if response.content:
                # Handle both string and list content
                if isinstance(response.content, str):
                    yield response.content
                elif isinstance(response.content, list):
                    for content_block in response.content:
                        if isinstance(content_block, dict) and content_block.get('type') == 'text':
                            yield content_block.get('text', '')
                        elif isinstance(content_block, str):
                            yield content_block
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"[Claude] Found {len(response.tool_calls)} tool call(s)")
                
                # Add the AI message with tool calls to history
                messages.append(response)
                
                # Execute each tool and collect results
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id')
                    
                    print(f"[Claude] Executing tool: {tool_name} with args: {tool_args}")
                    yield f"\n\n🔧 [Tool: {tool_name}({tool_args})]\n"
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool_obj in LANGCHAIN_TOOLS:
                        if tool_obj.name == tool_name:
                            tool_result = tool_obj.invoke(tool_args)
                            break
                    
                    if tool_result is None:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=json.dumps(tool_result),
                        tool_call_id=tool_id
                    ))
                
                yield "\n\n"
                # Continue loop to get final response
                continue
            else:
                # No tool calls, we're done
                break
                
    except Exception as e:
        print(f"[Claude] Error: {str(e)}")
        yield f"Error processing request: {str(e)}"

def gpt_models_stream(request: AI_Request, **kwargs) -> AsyncGenerator: # type: ignore
    """Handle all GPT model requests with dynamic model selection"""

    print(f"[GPT] Processing request for model: {request.model}")
    # Validate the model name
    
    if request.model not in valid_gpt_models:
        yield f"Error: Unsupported GPT model '{request.model}'"
        return

    try:
        # Initialize LangChain ChatOpenAI
        llm = ChatOpenAI(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True
        )
        
        # Check if this model supports tools
        supports_tools = is_tool_supported(request.model)
        
        # Bind tools to the model if supported
        if supports_tools:
            llm_with_tools = llm.bind_tools(LANGCHAIN_TOOLS)
            print(f"[GPT] Tools enabled for {request.model}")
        else:
            llm_with_tools = llm
            print(f"[GPT] No tool support for {request.model}")
        
        # System prompt for GPT models
        system_prompt = f"""
        You are {valid_gpt_models[request.model]}.
        You are a highly capable, thoughtful, and precise assistant.
        Your goal is to deeply understand the user's intent, ask clarifying questions when needed, think step-by-step through complex problems, provide clear and accurate answers, and proactively anticipate helpful follow-up information.
        Always prioritize being truthful, nuanced, insightful, and efficient, tailoring your responses specifically to the user's needs and preferences.
        
        # Core Principles
        - Provide the most useful, direct, and relevant answer possible.
        - Ask for clarification only when absolutely necessary to proceed.
        - State when you don’t know something or cannot perform a task.
        - Avoid making up facts, sources, or events. No hallucinating.
        - Never provide harmful, illegal, or dangerous instructions.
        - Respect privacy: do not generate personal data about real people.
        - Avoid disallowed content: hate, harassment, explicit sexual content involving minors, instructions for wrongdoing, etc.
        - Encourage safe, legal alternatives when denying a request.
        - Provide context, reasoning, and steps when useful.

        # Interaction Style
        - Default to a friendly, clear, direct tone.
        - Adapt writing style to the user’s preference (technical, casual, formal, humorous, etc.).
        - Avoid overly long or overly short answers—use the level of detail appropriate to the request.
        - When listing steps, be structured and easy to follow.
        - If asked for code, provide clean, minimal, correct examples.

        # Knowledge & Reasoning
        - Use reasoning explicitly when helpful but avoid exposing internal chain-of-thought.
        - Provide answers based on reliable information.
        - When uncertain, express uncertainty clearly.
        - If the user requests predictions or opinions, present them as speculative, not factual.

        # Tools
        - Use a tool only when the user asks for something requiring it.
        - State clearly when a tool does not support a requested capability.
        - Never fabricate tool output.
        """
        
        # Initialize conversation messages
        messages = [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=request.question)
        ]
        
        # Agent loop for handling tool calls
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"[GPT] Iteration {iteration}")
            
            # Use LangChain's message accumulation for proper tool call handling
            from langchain_core.messages import AIMessageChunk
            
            accumulated_message = None
            
            # Stream and accumulate chunks properly
            for chunk in llm_with_tools.stream(messages):
                # Yield content immediately for streaming UX
                if chunk.content:
                    yield chunk.content
                
                # Accumulate the message chunks properly
                if accumulated_message is None:
                    accumulated_message = chunk
                else:
                    accumulated_message = accumulated_message + chunk
            
            # Extract the complete response with properly formed tool calls
            response_content = accumulated_message.content if accumulated_message else ""
            tool_calls_list = accumulated_message.tool_calls if (accumulated_message and hasattr(accumulated_message, 'tool_calls')) else []
            
            # Check if there are tool calls
            if tool_calls_list:
                print(f"[GPT] Found {len(tool_calls_list)} tool call(s)")
                
                # Create AIMessage with tool calls
                ai_msg = AIMessage(
                    content=response_content,
                    tool_calls=tool_calls_list
                )
                messages.append(ai_msg)
                
                # Execute each tool
                for tool_call in tool_calls_list:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id')
                    
                    print(f"[GPT] Executing tool: {tool_name} with args: {tool_args}")
                    yield f"\n\n🔧 [Tool: {tool_name}({tool_args})]\n"
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool_obj in LANGCHAIN_TOOLS:
                        if tool_obj.name == tool_name:
                            tool_result = tool_obj.invoke(tool_args)
                            break
                    
                    if tool_result is None:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=json.dumps(tool_result),
                        tool_call_id=tool_id
                    ))
                
                yield "\n\n"
                # Continue loop to get final response
                continue
            else:
                # No tool calls, we're done
                break
                
    except Exception as e:
        print(f"[GPT] Error: {str(e)}")
        yield f"Error processing request: {str(e)}"

MODEL_FUNCTIONS = {
    "deepseek-chat": deepseek_chat_stream,
    "deepseek-reasoner": deepseek_reasoner_stream,
    "gpt-5.2": gpt_models_stream,
    "gpt-5.1": gpt_models_stream,
    "gpt-5-mini": gpt_models_stream,
    "gpt-5-nano": gpt_models_stream,
    "gpt-4.1": gpt_models_stream,
    "gpt-4.1-mini": gpt_models_stream,
    "gpt-4.1-nano": gpt_models_stream,
    "o4-mini-deep-research": gpt_models_stream,
    "claude-sonnet-4-5": claude_models_stream,
    "claude-haiku-4-5": claude_models_stream,
    "claude-opus-4-5": claude_models_stream,
    "claude-3-5-haiku-latest": claude_models_stream,
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