valid_gpt_models = {
  "gpt-5": "GPT-5 (Advanced reasoning and creativity with expert-level responses)",
  "gpt-5-mini": "GPT-5 Mini (Fast, efficient, and concise with solid reasoning)",
  "gpt-5-nano": "GPT-5 Nano (Ultra-light, delivers brief 1–2 sentence answers)",
  "gpt-4.1": "GPT-4.1 (Comprehensive answers with professional tone)",
  "gpt-4.1-mini": "GPT-4.1 Mini (Concise but informative responses)",
  "gpt-4.1-nano": "GPT-4.1 Nano (Very short 1-2 sentence answers)",
  "gpt-4o": "GPT-4o (Sophisticated, nuanced responses)",
}

valid_claude_models = {
  "claude-sonnet-4-5": "Claude Sonnet 4.5 (Next-gen Sonnet tier: high throughput, production workloads with strong coding & reasoning) ",
  "claude-haiku-4-5": "Claude Haiku 4.5 (Lightweight tier: fast responses, cost-efficient throughput, ideal for summaries, chat, and everyday tasks)",
  "claude-opus-4-5": "Claude Opus 4.1 (Top-tier flagship: complex reasoning, long context, full-agent capability and highest accuracy) ",
  "claude-3-5-haiku-latest": "Claude 3.5 Haiku (Fast and cost-efficient; good for everyday queries, moderation & translation)",
}

supports_tools = {
  "gpt-5",
  "gpt-4.1",
  "gpt-4o",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
  "claude-opus-4-5",
  "claude-3-5-haiku-latest", 
}

supports_tools_true_false = {
  "deepseek-reasoner" : "false",
  "deepseek-chat" : "false (for now)",
  "gpt-5" : "true",
  "gpt-5-mini" : "false",
  "gpt-5-nano" : "false",
  "gpt-4.1" : "true",
  "gpt-4.1-mini" : "false",
  "gpt-4.1-nano" : "false",
  "gpt-4o" : "true",
  "claude-sonnet-4-5": "true",
  "claude-haiku-4-5": "true",
  "claude-opus-4-5": "true",
  "claude-3-5-haiku-latest": "true",
  "claude-4-sonnet": "true",
}

valid_ollama_models = {
    "LLaMA 3.2 3B": "LLaMA 3.2 3B (Compact general-purpose model, good reasoning and instruction following)",
    "LLaMA 3.1 8B": "LLaMA 3.1 8B (Everyday chat, summarization, instruction-following, light coding)",
    "Qwen 2.5 7B": "Qwen 2.5 7B (Fast general-purpose chat, multilingual tasks, simple coding, structured outputs)",
    "Qwen 2.5 14B": "Qwen 2.5 14B (Complex reasoning, multi-step instructions, coding, structured outputs)",
    "Qwen 2.5 Coder 7B": "Qwen 2.5 Coder 7B (Coding-focused, code generation, debugging, refactoring, unit tests)",
    "Qwen 2.5 Coder 14B": "Qwen 2.5 Coder 14B (Advanced coding tasks, multi-file projects, detailed code explanation)",
    "Gemma 3 4B": "Gemma 3 4B (Fast, long-context, chat, summarization, creative writing, light coding)",
    "DeepSeek-R1 8B": "DeepSeek-R1 8B (Step-by-step reasoning, logic, planning, code analysis)",
    "DeepSeek Coder V2 16B": "DeepSeek Coder V2 16B (High-capacity coding, multi-file projects, advanced code tasks)",
    "Code LLaMa 7B": "Code LLaMa 7B (Coding, debugging, small-to-medium refactoring, single-file projects)",
    "Code LLaMa 13B": "Code LLaMa 13B (Advanced coding, multi-file projects, higher-quality code explanations)",
    "Mistral 7B": "Mistral 7B (Fast general-purpose chat, structured outputs, summarization, coding)",
    "LLaVA 7B": "LLaVA 7B (Multimodal vision-language, image understanding, screenshots, diagrams, basic chat)"
}

ollama_model_descriptions = {
    "LLaMA 3.2 3B": (
        ""
    ),
    "LLaMA 3.1 8B": (
        "Compact general-purpose model. Good for everyday chat, creative writing, summarization, "
        "instruction-following, and short code tasks. Fast inference on consumer GPUs, "
        "but struggles with very complex reasoning or extremely long context."
    ),
    "Qwen 2.5 7B": (
        "Efficient general-purpose model with strong multilingual support. Great for text chat, "
        "summaries, instruction-following, simple coding, and structured output generation. "
        "Fast and suitable for local consumer hardware."
    ),
    "Qwen 2.5 14B": (
        "Larger Qwen model for more complex reasoning, multi-step instructions, and longer context. "
        "Good at general chat, coding, structured outputs, and translation. Slower than 7B but more capable."
    ),
    "Qwen 2.5 Coder 7B": (
        "Coding-specialized 7B model. Excels at code generation, debugging, refactoring, "
        "unit test creation, and pseudocode conversion. Best for small-to-medium coding tasks."
    ),
    "Qwen 2.5 Coder 14B": (
        "Larger coding-focused model. Handles more complex codebases, multi-file projects, "
        "code explanation, and reasoning tasks. Slower than 7B but more reliable for complex tasks."
    ),
    "Gemma 3 4B": (
        "Modern, lightweight model optimized for speed and long-context handling. "
        "Good for chat, summarization, creative writing, instruction-following, "
        "and light coding. Excellent performance on consumer GPUs."
    ),
    "DeepSeek-R1 8B": (
        "Reasoning-specialized model. Excels at chain-of-thought problems, logic puzzles, "
        "step-by-step reasoning, planning, and code reasoning. Great for structured reasoning tasks."
    ),
    "DeepSeek Coder V2 16B": (
        "High-capacity coding-focused model. Excellent for multi-file projects, complex refactors, "
        "unit test generation, and advanced code explanation. Requires significant GPU resources."
    ),
    "Code LLaMa 7B": (
        "Coding-focused LLaMA variant. Good for code generation, debugging, and small-to-medium "
        "refactoring tasks. Reliable for single-file projects and coding exercises."
    ),
    "Code LLaMa 13B": (
        "Larger coding LLaMA variant. Handles more complex projects, multi-file codebases, "
        "and higher-quality code explanation. Slower on consumer GPUs but more capable."
    ),
    "Mistral 7B": (
        "High-quality, efficient 7B general-purpose model. Excellent for chat, structured output, "
        "summarization, coding, and instruction-following. Fast and lightweight on consumer GPUs."
    ),
    "LLaVA 7B": (
        "Multimodal Vision-Language model. Excels at interpreting images, screenshots, diagrams, "
        "and combining visual input with text reasoning. Also capable of basic text-only tasks."
    )
}

systemPrompts = {
  "chatgpt": (
    f"""
    You are <insert model here>.
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
  ),
  "Claude": (
    f"""
    You are <insert model here>.
    Provide the best possible answer to the user's question.
    
    # Tone and style
    - You should be concise, direct, and to the point.
    - You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.
    - IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
    - IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.

    # Proactiveness
    You are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:
    - Doing the right thing when asked, including taking actions and follow-up actions
    - Not surprising the user with actions you take without asking
    For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.

    # Following conventions
    When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
    - NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).
    - When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.
    - When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.
    - Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.

    # Code style
    - IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked
    """
  ),
}