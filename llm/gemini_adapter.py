import os
import requests
from typing import Any, Dict

from llm.base_adapter import LLMAdapter
from config.logging_config import logger

class GeminiModelAdapter(LLMAdapter):
    """
    Concrete implementation of the LLMAdapter for the Gemini API backend using direct HTTP POST requests.
    """

    def __init__(self, search_system: Any, model_name: str):
        super().__init__(search_system, model_name)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.is_ready = False

        self.system_prompt = self.system_prompt.strip()

    def setup(self) -> bool:
        """
        Verifies the Gemini API key and connectivity to the REST endpoint.
        """
        logger.info(f"🚀 Setting up Gemini API (HTTP) with model: {self.model_name}")

        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY environment variable not found.")
            self.is_ready = False
            return False

        # We don't have a specific endpoint to "verify" the model name via REST,
        # so we'll just mark it as ready.
        self.is_ready = True
        logger.info("✅ Gemini API key found. Ready to send requests.")
        return True

    def ask(self, prompt: str, max_token: int = 8096, temp: float = 0.2) -> tuple[str, dict]:
        """
        Public method for the Bridge to call for direct LLM generation.
        """
        if not self.is_ready:
            logger.error(f"Gemini model '{self.model_name}' is not set up or ready.")
            raise RuntimeError(f"Gemini model '{self.model_name}' is not set up or ready.")
        return self._generate(prompt, max_token, temp)

    def _generate(self, prompt: str, max_token: int = 8096, temp: float = 0.2) -> tuple[str, dict]:
        """
        Core implementation: Sends the prompt to the Gemini API using a REST POST request.
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_token
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Extract text safely
            candidates = data.get("candidates", [])
            if not candidates:
                raise Exception("API returned no candidates.")

            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            finish_reason = candidates[0].get("content", {}).get("finishReason", "STOP")
            if finish_reason != "STOP":
                logger.error(f"API returned unexpected finish reason: {finish_reason}")
                raise Exception(f"API returned unexpected finish reason: {finish_reason}")
            if not text:
                logger.error("Empty response text from Gemini API.")
                raise Exception("Empty response text from Gemini API.")

            # Optionally print token usage if available
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", None)
            candidates_token_count = usage.get("candidatesTokenCount", None)
            thoughts_token_count = usage.get("thoughtsTokenCount", None)
            total_token_count = usage.get("totalTokenCount", None)
            token_count = {'promptTokenCount': prompt_tokens, 'candidatesTokenCount': candidates_token_count,
                           'thoughtsTokenCount': thoughts_token_count, 'totalTokenCount': total_token_count}
            if usage:
                logger.info(f"🔹 Prompt Tokens: {prompt_tokens}")
                logger.info(f"🔹 Candidate Tokens: {candidates_token_count}")
                logger.info(f"🔹 Thoughts Token: {thoughts_token_count}")
                logger.info(f"🔹 Total Token: {total_token_count}")

            return text, token_count

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed: {e}")
            raise Exception(f"HTTP Request failed: {e}")
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise Exception(f"Gemini generation error: {e}")
