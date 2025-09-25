from typing import Dict, Any

from llm.base_adapter import LLMAdapter
from llm.config import LLMConfig
from llm.ollama_adapter import OllamaModelAdapter


class LocalLLMBridge:
    """
    The main generic class to manage local LLM models and RAG operations.
    Uses an Adapter pattern for future extensibility.
    """

    # Store the dictionary of available adapters for easy extension
    AVAILABLE_ADAPTERS = {
        "ollama": OllamaModelAdapter,
        # In the future, you would add:
        # "transformers": TransformersModelAdapter,
    }

    def __init__(self, search_system: Any, model_key: str, backend_type: str = "ollama"):
        """
        Initializes the bridge by selecting and instantiating the correct model adapter.

        :param search_system: The external search/vector-db system instance.
        :param model_key: The simplified key of the model (e.g., 'phi3.5_q4_K_M').
        :param backend_type: The backend to use (e.g., 'ollama'). Defaults to 'ollama'.
        """

        if backend_type not in self.AVAILABLE_ADAPTERS:
            raise ValueError(
                f"Unknown backend type: {backend_type}. Must be one of: {list(self.AVAILABLE_ADAPTERS.keys())}")

        if model_key not in LLMConfig.RECOMMENDED_MODELS.get(backend_type, {}):
            raise ValueError(f"Model key '{model_key}' not found for backend '{backend_type}'.")

        model_name = LLMConfig.RECOMMENDED_MODELS[backend_type][model_key]["name"]
        AdapterClass = self.AVAILABLE_ADAPTERS[backend_type]

        # Instantiate the specific adapter (e.g., OllamaModelAdapter)
        self.adapter: LLMAdapter = AdapterClass(search_system, model_name)
        self.backend_type = backend_type
        self.model_key = model_key
        print(f"⚙️ Bridge created for {self.backend_type} using model: {model_name}")

    def setup_model(self) -> bool:
        """Sets up the underlying model/backend (e.g., pulling the Ollama model)."""
        return self.adapter.setup()

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Performs the RAG query via the active adapter."""
        return self.adapter.ask(question, top_k=top_k)