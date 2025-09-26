from typing import Dict, Any

from llm.base_adapter import LLMAdapter
from llm.config import LLMConfig
from llm.ollama_adapter import OllamaModelAdapter


class LocalLLMBridge:
    """
    Manages local Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) operations.

    This class serves as a central point for interacting with different local LLM backends.
    It utilizes the Adapter pattern, which allows for easy integration of new backends
    (e.g., Hugging Face Transformers, Llama.cpp) without changing the core logic.
    """

    # A class-level dictionary to store available adapters.
    # This design makes it simple to add new LLM backends in the future.
    AVAILABLE_ADAPTERS = {
        "ollama": OllamaModelAdapter,
        # Future adapters can be added here, for example:
        # "transformers": TransformersModelAdapter,
        # "llama_cpp": LlamaCppAdapter,
    }

    def __init__(self, search_system: Any, model_key: str, backend_type: str = "ollama"):
        """
        Initializes the bridge by selecting and instantiating the correct model adapter.

        Args:
            search_system (Any): The external search/vector-db system instance,
                                 used by the adapter for RAG.
            model_key (str): A simplified key for the model (e.g., 'phi3.5_q4_K_M').
                             This key is used to look up the full model name in the configuration.
            backend_type (str): The type of LLM backend to use (e.g., 'ollama').
                                Defaults to 'ollama'.
        
        Raises:
            ValueError: If the `backend_type` is not in `AVAILABLE_ADAPTERS` or
                        if the `model_key` is not found for the specified backend.
        """

        # Validate that the requested backend type is supported.
        if backend_type not in self.AVAILABLE_ADAPTERS:
            raise ValueError(
                f"Unknown backend type: {backend_type}. Must be one of: {list(self.AVAILABLE_ADAPTERS.keys())}")

        # Validate that the requested model key exists for the chosen backend.
        if model_key not in LLMConfig.RECOMMENDED_MODELS.get(backend_type, {}):
            raise ValueError(f"Model key '{model_key}' not found for backend '{backend_type}'.")

        # Retrieve the full model name from the configuration using the provided key.
        model_name = LLMConfig.RECOMMENDED_MODELS[backend_type][model_key]["name"]
        
        # Get the class of the adapter from the AVAILABLE_ADAPTERS dictionary.
        AdapterClass = self.AVAILABLE_ADAPTERS[backend_type]

        # Instantiate the specific adapter class (e.g., `OllamaModelAdapter`)
        # and store it as the main active adapter. The type hint `LLMAdapter`
        # ensures consistency.
        self.adapter: LLMAdapter = AdapterClass(search_system, model_name)
        
        # Store the configuration details for future reference if needed.
        self.backend_type = backend_type
        self.model_key = model_key
        print(f"⚙️ Bridge created for {self.backend_type} using model: {model_name}")

    def setup_model(self) -> bool:
        """
        Sets up the underlying model and backend.

        This method delegates the setup process to the active adapter.
        For example, with Ollama, this might involve pulling the model from the registry.

        Returns:
            bool: True if the setup was successful, False otherwise.
        """
        return self.adapter.setup()

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """
        Performs a Retrieval-Augmented Generation (RAG) query.

        This method forwards the user's question to the active adapter's `ask` method.
        The adapter handles the full RAG workflow, including document retrieval and
        prompt generation.

        Args:
            question (str): The user's question.
            top_k (int): The number of top relevant documents to retrieve for context.

        Returns:
            Dict: The response from the LLM, typically containing the answer and source documents.
        """
        return self.adapter.ask(question, top_k=top_k)
