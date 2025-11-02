import os
import requests
from typing import Any, Dict

from llm.base_adapter import LLMAdapter


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
        print(f"🚀 Setting up Gemini API (HTTP) with model: {self.model_name}")

        if not self.api_key:
            print("❌ GEMINI_API_KEY environment variable not found.")
            self.is_ready = False
            return False

        # We don't have a specific endpoint to "verify" the model name via REST,
        # so we'll just mark it as ready.
        self.is_ready = True
        print("✅ Gemini API key found. Ready to send requests.")
        return True

    def ask(self, prompt: str, max_token: int = 8096, temp: float = 0.2) -> str:
        """
        Public method for the Bridge to call for direct LLM generation.
        """
        if not self.is_ready:
            raise RuntimeError(f"Gemini model '{self.model_name}' is not set up or ready.")
        return self._generate(prompt, max_token, temp)

    def _generate(self, prompt: str, max_token: int = 8096, temp: float = 0.2) -> str:
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

        # Add system instruction if available

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
                raise Exception(f"API returned unexpected finish reason: {finish_reason}")
            if not text:
                raise Exception("Empty response text from Gemini API.")

            # Optionally print token usage if available
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", None)
            candidatesTokenCount = usage.get("candidatesTokenCount", None)
            thoughtsTokenCount = usage.get("thoughtsTokenCount", None)
            totalTokenCount = usage.get("totalTokenCount", None)
            if usage:
                print(f"🔹 Prompt Tokens: {prompt_tokens}")
                print(f"🔹 Candidate Tokens: {candidatesTokenCount}")
                print(f"🔹 Thoughts Token: {thoughtsTokenCount}")
                print(f"🔹 Total Token: {totalTokenCount}")

            return text

        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP Request failed: {e}")
        except Exception as e:
            raise Exception(f"Gemini generation error: {e}")

if __name__ == "__main__":
    # Example usage
    gemini_adapter = GeminiModelAdapter("gemini-pro", "gemini-2.5-pro")
    gemini_adapter.setup()
    print(gemini_adapter.ask("What is the capital of France ?"))