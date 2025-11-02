class LLMConfig:
    """Regroups all model configurations and acts as the central model directory."""

    # NOTE: The version ONLY includes 'ollama' models for now
    AVAILABLE_MODELS = {
        "phi3.5_q4_K_M": {
            "model_backend": "ollama",
            "name": "phi3.5:3.8b-mini-instruct-q4_K_M",
            "size": "2.2GB",
            "description": "Microsoft Phi-3.5 - Excellent for RAG, very fast",
            "ram_needed": "4GB",
            "best_for": "General Q&A, great instruction following"
        },
        "phi3.5_q8_0": {
            "model_backend": "ollama",
            "name": "phi3.5:3.8b-mini-instruct-q8_0",
            "size": "4.1GB",
            "description": "Microsoft Phi-3.5 Q8_0 - Highest performance, heavy resource usage",
            "ram_needed": "8GB",
            "best_for": "Maximum quality responses, complex reasoning, long documents"
        },
        "phi3.5_q6_K":{
            "model_backend": "ollama",
            "name": "phi3.5:3.8b-mini-instruct-q6_K",
            "size": "3.1GB",
            "description": "Microsoft Phi-3.5 Q6_K - Enhanced performance, more memory",
            "ram_needed": "6GB",
            "best_for": "Longer context tasks, instruction-following with higher accuracy"
        },
        "llama3.2": {
            "model_backend": "ollama",
            "name": "llama3.2:3b-instruct-q4_K_M",
            "size": "1.9GB",
            "description": "Meta Llama 3.2 - Great balance of size/quality",
            "ram_needed": "3GB",
            "best_for": "Conversational, good reasoning"
        },
        "qwen2.5": {
            "model_backend": "ollama",
            "name": "qwen2.5:3b-instruct-q4_K_M",
            "size": "1.9GB",
            "description": "Alibaba Qwen2.5 - Excellent for technical content",
            "ram_needed": "3GB",
            "best_for": "Technical docs, coding, structured responses"
        },
        "pleias-rag-350m": {
            "model_backend": "transformers",
            "name": "PleIAs/Pleias-RAG-350M",
            "size": "",
            "description": "",
            "ram_needed": "",
            "best_for": ""
        },
        "mistral-7B-instruct": {
            "model_backend": "transformers",
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "size": "",
            "description": "",
            "ram_needed": "",
            "best_for": ""
        },
        "gemma2": {
            "model_backend": "ollama",
            "name": "gemma2:2b-instruct-q4_K_M",
            "size": "1.6GB",
            "description": "Google Gemma2 - Very small but capable",
            "ram_needed": "2GB",
            "best_for": "Resource-constrained environments"
        },
        "flash": {
            "model_backend": "gemini",
            "name": "gemini-2.5-flash",
            "context_window": "1,048,576 tokens",
            "latency_priority": "Low Latency / High Throughput",
            "description": "The best price-performance model. Fast, cost-effective, and capable of handling complex RAG with a 1M token context window.",
            "best_for": "Large-scale RAG, high-volume tasks, agentic workflows"
        },
        "pro": {
            "model_backend": "gemini",
            "name": "gemini-2.5-pro",
            "context_window": "1,048,576 tokens",
            "latency_priority": "Standard Latency / High Quality",
            "description": "Google's most advanced reasoning model, featuring the highest quality output and complex problem-solving capabilities.",
            "best_for": "Complex reasoning, coding, deep analysis, highest quality RAG"
        },
        "flash_lite": {
            "model_backend": "gemini",
            "name": "gemini-2.5-flash-lite",
            "context_window": "1,048,576 tokens",
            "latency_priority": "Very Low Latency / Most Cost-Effective",
            "description": "Optimized for maximum speed and cost-efficiency. A lightweight model suitable for high-frequency or cost-sensitive tasks.",
            "best_for": "High throughput, cost-conscious applications, simple Q&A"
        },
        "flash_1_0": {
            "model_backend": "gemini",
            "name": "gemini-1.0-flash",
            "context_window": "1,048,576 tokens",
            "latency_priority": "Standard Latency / Legacy Option",
            "description": "A previous generation model that offers a stable, general-purpose API option.",
            "best_for": "Legacy compatibility, general tasks"
        }
    }
