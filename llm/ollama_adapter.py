# File: llm/ollama_adapter.py

import subprocess
import time
from typing import Any

import requests

from config.logging_config import logger
from llm.base_adapter import LLMAdapter


class OllamaModelAdapter(LLMAdapter):
    """
    Concrete implementation of the LLMAdapter for the Ollama backend.

    This adapter manages the lifecycle of an Ollama model, including
    checking its status, ensuring it is pulled, and handling API
    interactions for basic generation. The RAG orchestration logic is 
    delegated to the LocalLLMBridge.
    """

    def __init__(self, search_system: Any, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initializes the Ollama adapter.

        Args:
            search_system (Any): The retrieval/search system instance for RAG (stored but not used by this adapter).
            model_name (str): The specific name of the model to use (e.g., 'llama3:8b').
            base_url (str): The URL where the Ollama server is running (default is local).
        """
        super().__init__(search_system, model_name)
        self.base_url = base_url  # The endpoint for the Ollama API
        self.is_ready = False  # Flag to track if the model is set up and running

    def check_ollama_status(self) -> bool:
        """
        Checks if the Ollama server is running and if the required model is installed/available.
        Returns:
            bool: True if the server is responsive and the model is found, False otherwise.
        """
        try:
            # Attempt to hit the Ollama API tags endpoint to get installed models
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code != 200:
                logger.warning(f"Ollama server responded with status code: {response.status_code}")
                return False

            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            # Check if the specific model is installed. The check is flexible (e.g., checks for 'phi3' in 'phi3:3.8b').
            return any(self.model_name in name for name in model_names)
        except requests.exceptions.RequestException as e:
            # Handles connection errors (e.g., server not running)
            logger.debug(f"Ollama connection error: {e}")
            return False
        except Exception as e:
            # Catches other potential issues (e.g., JSON decoding error)
            logger.error(f"Unexpected error during Ollama status check: {e}")
            return False

    def setup(self) -> bool:
        """
        Sets up the Ollama environment: checks for CLI, starts the server (if necessary), and pulls the model.
        This implementation attempts to handle the local setup process for a typical Ollama installation.
        Returns:
            bool: True if the model is successfully confirmed ready, False otherwise.
        """
        print(f"🚀 Setting up Ollama with model: {self.model_name}")

        # 1. Check for 'ollama' CLI executable
        try:
            # Run a command to check if the `ollama` executable is in the PATH
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            print("✅ Ollama is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama not found. Please install from: https://ollama.ai")
            return False

        # 2. Check server status and attempt to start if not running
        if not self.check_ollama_status():
            print("🔄 Attempting to start/wait for Ollama server...")
            try:
                # Start the server in the background using `ollama serve`.
                # Note: `subprocess.Popen` is non-blocking and relies on the user's system to keep it running.
                subprocess.Popen(["ollama", "serve"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                time.sleep(3)  # Give the server a moment to spin up
            except Exception as e:
                print(f"❌ Failed to start Ollama server: {e}")

        # 3. Install the model if still not present after server check/start
        if not self.check_ollama_status():
            print(f"📥 Installing model: {self.model_name}. This may take a few minutes...")
            try:
                # Use `ollama pull` to download and install the model
                result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True, text=True, timeout=600  # 10 minutes timeout for pull
                )
                if result.returncode == 0:
                    print("✅ Model installed successfully!")
                else:
                    # Log the last line of stderr as the error message
                    error_message = result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'
                    print(
                        f"❌ Failed to install model: {error_message}")
                    return False
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"❌ Error installing model: {e}")
                return False

        # Final status check to confirm readiness
        self.is_ready = self.check_ollama_status()
        return self.is_ready

    def ask(self, prompt: str,  max_token: int = 500, temp: float = 0.2) -> str:
        """
        Public method for the Bridge to call for direct LLM generation.
        Delegates to the internal _generate method.
        """
        if not self.is_ready:
            raise RuntimeError(f"Ollama model '{self.model_name}' is not set up or ready.")

        return self._generate(prompt, max_token, temp)

    def _generate(self, prompt: str, max_token: int = 500, temp: float = 0.2) -> str:
        """
        Abstract core method implementation: sends a prompt to the Ollama /api/generate endpoint.
        Args:
            prompt (str): The final, formatted prompt to send to the LLM.
        Returns:
            str: The raw text response from the model.
        Raises:
            Exception: If the API call fails or returns a non-200 status code.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # Request the full response at once
            "options": {"temperature": temp, "num_predict": max_token}
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            # Extract and strip the final generated response text
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama API Error: HTTP {response.status_code} - {response.text}")
