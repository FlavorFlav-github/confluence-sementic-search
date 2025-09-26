import subprocess
import time
from typing import Dict, Any

import requests

from config.logging_config import logger
from llm.base_adapter import LLMAdapter
from config.settings import ENRICH_WITH_NEIGHBORS


class OllamaModelAdapter(LLMAdapter):
    """Concrete implementation for the Ollama backend."""

    def __init__(self, search_system: Any, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(search_system, model_name)
        self.base_url = base_url
        self.is_ready = False

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            # Check if the specific model or a similar one is installed (e.g., llama3.2 in llama3.2:tag)
            return any(self.model_name.split(':')[0] in name for name in model_names)
        except requests.exceptions.RequestException:
            return False
        except Exception:
            return False

    def setup(self) -> bool:
        """Setup Ollama and install the model (simplified version of original logic)."""
        print(f"🚀 Setting up Ollama with model: {self.model_name}")

        # 1. Check for 'ollama' CLI (using subprocess)
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            print("✅ Ollama is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama not found. Please install from: https://ollama.ai")
            return False

        # 2. Start server/wait for status
        if not self.check_ollama_status():
            print("🔄 Attempting to start/wait for Ollama server...")
            try:
                # Start server in background (NOTE: This might need platform-specific handling in a real app)
                subprocess.Popen(["ollama", "serve"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                time.sleep(3)  # Wait for server to start
            except Exception as e:
                print(f"❌ Failed to start Ollama server: {e}")

        # 3. Install the model if not present (using subprocess pull)
        if not self.check_ollama_status():
            print(f"📥 Installing model: {self.model_name}. This may take a few minutes...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    print("✅ Model installed successfully!")
                else:
                    print(
                        f"❌ Failed to install model: {result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'}")
                    return False
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"❌ Error installing model: {e}")
                return False

        self.is_ready = self.check_ollama_status()
        return self.is_ready

    def _generate(self, prompt: str) -> str:
        """Generate response using Ollama API (internal implementation)."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 500}  # Simplified options
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama API Error: HTTP {response.status_code} - {response.text}")

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Performs RAG process using the Ollama model (retains original logic)."""
        if not self.is_ready:
            raise RuntimeError("Ollama model is not set up or ready.")

        # Step 0: Refine query (using local LLM)
        try:
            refine_prompt = f"Rewrite '\"{question}\"' into 3 concise, alternative search queries. Return them as a bullet list, without explanation."
            refined_output = self._generate(refine_prompt)
            refined_queries = [q.strip("-• ").strip() for q in refined_output.splitlines() if q.strip()]
            refined_queries.insert(0, question)
            logger.info(f"🔍 Refined queries: {refined_queries}")
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            refined_queries = [question]

        search_results = self.search.semantic_search(refined_queries, top_k=top_k)
        if not search_results:
            return {'question': question, 'answer': "I couldn't find any relevant information.", 'sources': [],
                    'model_used': self.model_name}

        enriched_results = []
        seen = set()
        for result in search_results:
            if (result.page_id, result.chunk_id) not in seen:
                enriched_results.append(result)
                seen.add((result.page_id, result.chunk_id))

            neighbors = self.search.fetch_adjacent_chunks(result, k=ENRICH_WITH_NEIGHBORS)
            for n in neighbors:
                if (n.page_id, n.chunk_id) not in seen:
                    enriched_results.append(n)
                    seen.add((n.page_id, n.chunk_id))

        context_pieces = []
        for i, result in enumerate(enriched_results, 1):
            context_pieces.append(
                {'text': f"[Source {i} - {result.title}]\n{result.text.strip()}",
                 'title': result.title,
                 'link': result.link,
                 'score': result.score}
            )

        context = "\n\n".join([piece['text'] for piece in context_pieces])

        final_prompt = f"""DOCUMENTATION:\n{context}\n\nQUESTION: {question}"""

        # Step 4: Generate with local LLM
        try:
            print("🤖 Generating answer with local LLM...")
            answer = self._generate(final_prompt)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            logger.error(f"LLM generation error: {e}")

        return {
            'question': question,
            'answer': answer,
            'sources': [{'title': piece['title'], 'link': piece['link'], 'score': piece['score']} for piece in
                        context_pieces],
            'model_used': self.model_name
        }
