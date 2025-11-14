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
  "claude-opus-4-1": "Claude Opus 4.1 (Top-tier flagship: complex reasoning, long context, full-agent capability and highest accuracy) ",
  "claude-3-5-haiku-latest": "Claude 3.5 Haiku (Fast and cost-efficient; good for everyday queries, moderation & translation)",
  "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet (Balanced performance and cost; capable of deeper reasoning & data tasks)",
  "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet (Hybrid reasoning mode: choose speed vs depth; advanced coding and agent workflows) ",
}

supports_tools = {
  "deepseek-chat",
  "gpt-5",
  "gpt-4.1",
  "gpt-4o",
  "claude-sonnet-4-5",
  "claude-opus-4-1",
  "claude-3-5-haiku-latest",
  "claude-3-5-sonnet-latest",
  "claude-3-7-sonnet-latest",
}

supports_tools_true_false = {
  "deepseek-reasoner" : "false",
  "deepseek-chat" : "true",
  "gpt-5" : "true",
  "gpt-5-mini" : "false",
  "gpt-5-nano" : "false",
  "gpt-4.1" : "true",
  "gpt-4.1-mini" : "false",
  "gpt-4.1-nano" : "false",
  "gpt-4o" : "true",
  "claude-sonnet-4-5": "true",
  "claude-opus-4-1": "true",
  "claude-3-5-haiku-latest": "true",
  "claude-3-5-sonnet-latest": "true",
  "claude-3-7-sonnet-latest": "true",
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