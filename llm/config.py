class LLMConfig:
    """Regroups all model configurations and acts as the central model directory."""

    # NOTE: The version ONLY includes 'ollama' models for now, as requested.
    RECOMMENDED_MODELS = {
        "ollama": {
            "phi3.5_q4_K_M": {
                "name": "phi3.5:3.8b-mini-instruct-q4_K_M",
                "size": "2.2GB",
                "description": "Microsoft Phi-3.5 - Excellent for RAG, very fast",
                "ram_needed": "4GB",
                "best_for": "General Q&A, great instruction following"
            },
            "phi3.5_q8_0": {
                "name": "phi3.5:3.8b-mini-instruct-q8_0",
                "size": "4.1GB",
                "description": "Microsoft Phi-3.5 Q8_0 - Highest performance, heavy resource usage",
                "ram_needed": "8GB",
                "best_for": "Maximum quality responses, complex reasoning, long documents"
            },
            "phi3.5_q6_K":{
                "name": "phi3.5:3.8b-mini-instruct-q6_K",
                "size": "3.1GB",
                "description": "Microsoft Phi-3.5 Q6_K - Enhanced performance, more memory",
                "ram_needed": "6GB",
                "best_for": "Longer context tasks, instruction-following with higher accuracy"
            },
            "llama3.2": {
                "name": "llama3.2:3b-instruct-q4_K_M",
                "size": "1.9GB",
                "description": "Meta Llama 3.2 - Great balance of size/quality",
                "ram_needed": "3GB",
                "best_for": "Conversational, good reasoning"
            },
            # ... all other ollama models from the original class ...
            "qwen2.5": {
                "name": "qwen2.5:3b-instruct-q4_K_M",
                "size": "1.9GB",
                "description": "Alibaba Qwen2.5 - Excellent for technical content",
                "ram_needed": "3GB",
                "best_for": "Technical docs, coding, structured responses"
            },
            "gemma2": {
                "name": "gemma2:2b-instruct-q4_K_M",
                "size": "1.6GB",
                "description": "Google Gemma2 - Very small but capable",
                "ram_needed": "2GB",
                "best_for": "Resource-constrained environments"
            }
        },
        # Future models (like 'transformers') are hidden/commented for now,
        # fulfilling the requirement "the version should only include ollama models for now"
        # "transformers": {
        #     "phi3_mini": {
        #         "name": "microsoft/Phi-3-mini-4k-instruct",
        #         "size": "7.6GB",
        #         "description": "Direct Hugging Face integration",
        #         "ram_needed": "8GB",
        #         "best_for": "Full control, no external dependencies"
        #     }
        # }
    }

    @classmethod
    def print_recommendations(cls):
        """Prints a simplified recommendation overview."""
        print("🤖 RECOMMENDED LOCAL LLMs FOR RAG (Ollama):")
        print("=" * 60)
        # Use existing logic for recommendations (you can expand this with more detailed logic if needed)
        print("\n🥇 BEST OVERALL (Recommended):")
        print("   Model: Phi-3.5 Mini (3.8B parameters) - q4_K_M")
        print("   Size: 2.2GB | RAM: 4GB | Speed: Very Fast")
        print("   Why: Optimized for instruction following and RAG.")

        print("\n🥈 MOST EFFICIENT:")
        print("   Model: Gemma2 2B")
        print("   Size: 1.6GB | RAM: 2GB | Speed: Fastest")
        print("   Why: Smallest but still very capable for Q&A.")
        print("-" * 60)