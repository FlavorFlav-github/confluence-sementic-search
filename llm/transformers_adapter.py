# File: llm/transformer_adapter.py

from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from config.logging_config import logger
from llm.base_adapter import LLMAdapter


class TransformerModelAdapter(LLMAdapter):
    """
    Concrete implementation of the LLMAdapter for Hugging Face transformer models.

    This adapter handles setup, context retrieval, and inference
    for models such as Pleias-RAG-350M or Pleias-Pico.
    """

    def __init__(self, search_system: Any, model_name: str, device: str = None):
        """
        Initializes the Transformer adapter.

        Args:
            search_system (Any): Retrieval system instance (used for RAG context fetching).
            model_name (str): Name/path of the Hugging Face model (e.g., "PleIAs/Pleias-RAG-350M").
            device (str, optional): Device identifier ('cuda', 'cpu', etc.). Defaults to GPU if available.
        """
        super().__init__(search_system, model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.is_ready = False

    # ---------------------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------------------
    def setup(self) -> bool:
        """
        Loads the model and tokenizer from Hugging Face and prepares the inference pipeline.
        Returns:
            bool: True if the model is successfully loaded and ready.
        """
        try:
            logger.info(f"🚀 Loading Hugging Face model '{self.model_name}' on device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Try causal LM first (for LLaMA-like models)
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                model_type = "causal"
            except Exception as e:
                logger.warning(f"⚠️ Not a causal LM: {e}, trying Seq2Seq model...")
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if "cuda" in self.device else torch.float32
                ).to(self.device)
                model_type = "seq2seq"

            # Create appropriate pipeline
            task = "text-generation" if model_type == "causal" else "text2text-generation"
            self.generator = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if "cuda" in self.device else -1
            )

            self.is_ready = True
            logger.info(f"✅ Loaded {model_type.upper()} model '{self.model_name}' successfully.")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading transformer model '{self.model_name}': {e}")
            self.is_ready = False
            return False

    # ---------------------------------------------------------------------
    # ASK (RAG ORCHESTRATION)
    # ---------------------------------------------------------------------
    def ask(self, prompt: str, max_token: int = 500, temp: float = 0.2) -> Dict:
        """
        Performs a RAG-based answer generation using the local transformer model.

        Args:
            prompt (str): The fully formated prompt including question, context and sources.
            max_token (int): Maximum number of tokens for generation.
            temp (float): Temperature for creativity/randomness.

        Returns:
            Dict: {
                "answer": str,
                "context": List[str],
                "prompt": str
            }
        """
        if not self.is_ready:
            raise RuntimeError(f"Transformer model '{self.model_name}' not initialized. Call setup() first.")

        return self._generate(prompt, max_token, temp)

    # ---------------------------------------------------------------------
    # _GENERATE (LOW-LEVEL MODEL CALL)
    # ---------------------------------------------------------------------
    def _generate(self, prompt: str, max_token: int = 500, temp: float = 0.2) -> str:
        """
        Sends the constructed prompt to the transformer model for inference.

        Args:
            prompt (str): Full text prompt (with context and instructions).
            max_token (int): Max number of tokens for generation.
            temp (float): Sampling temperature.

        Returns:
            str: Generated model output.
        """
        try:
            outputs = self.generator(
                prompt,
                max_length=max_token,
                do_sample=True,
                temperature=temp
            )
            response = outputs[0]["generated_text"].strip()
            print(response)
            return response

        except Exception as e:
            logger.error(f"❌ Generation error for '{self.model_name}': {e}")
            return "Error generating response."

