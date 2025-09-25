from abc import abstractmethod, ABC
from typing import Any, Dict


class LLMAdapter(ABC):
    """Abstract Base Class for any LLM backend (Ollama, Transformers, etc.)."""

    def __init__(self, search_system: Any, model_name: str):
        self.search = search_system
        self.model_name = model_name
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided documentation.
        INSTRUCTIONS:
        - Answer based ONLY on the provided context
        - Be concise but comprehensive
        - If information is missing, say so clearly
        - Use bullet points or numbered lists when appropriate"""

    @abstractmethod
    def setup(self) -> bool:
        """Setup the backend and install/load the model."""
        pass

    @abstractmethod
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Ask a question, perform RAG, and get an answer."""
        pass

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Core method to interact with the specific LLM backend."""
        pass
