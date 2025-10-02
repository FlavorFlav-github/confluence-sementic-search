"""
API Layer for Confluence RAG system.
Exposes endpoints for asking questions against the Confluence index.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import your modules
from config.logging_config import logger
from config.settings import LLM_MODEL_GENERATION, LLM_MODEL_REFINE, \
    LLM_BACKEND_TYPE_GENERATION, LLM_BACKEND_TYPE_REFINEMENT, DEFAULT_TOP_K, RERANK_TOP_K, SOURCE_THRESHOLD
from indexer.hybrid_index import HybridSearchIndex
from indexer.qdrant_utils import check_and_start_qdrant
from llm.bridge import LocalLLMBridge
from llm.config import LLMConfig
from search.advanced_search import AdvancedSearch


# ---------------------------
# Request / Response Schemas
# ---------------------------

class QuestionRequest(BaseModel):
    question: str
    model: str
    search_top_k: Optional[int] = DEFAULT_TOP_K
    search_min_score: Optional[float] = SOURCE_THRESHOLD
    llm_top_k: Optional[int] = RERANK_TOP_K


class Source(BaseModel):
    title: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


# ---------------------------
# App Initialization
# ---------------------------

app = FastAPI(title="Confluence RAG API", version="1.0.0")

logger.info("Initializing Confluence RAG API...")

try:
    qdrant = check_and_start_qdrant()
    hybrid_search_index = HybridSearchIndex()
    hybrid_search_index.load_tfidf()
    search_system = AdvancedSearch(qdrant, hybrid_search_index)

    # Print recommendations (nice to have)
    LLMConfig.print_recommendations()

except Exception as e:
    logger.error(f"Failed to initialize RAG API: {e}")
    rag_system = None
    raise


# ---------------------------
# Endpoints
# ---------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    rag_system = LocalLLMBridge(
        search_system=search_system,
        generation_model_key=LLM_MODEL_GENERATION,
        refinement_model_key=LLM_MODEL_REFINE,
        generation_model_backend_type=LLM_BACKEND_TYPE_GENERATION,
        refinement_model_backend_type=LLM_BACKEND_TYPE_REFINEMENT
    )
    rag_system.setup_model()
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        result = rag_system.ask(request.question,
                                top_k=request.search_top_k,
                                final_top_k=request.llm_top_k,
                                score_threshold=request.search_min_score)

        # Clean the answer formatting
        answer_text = result.get("answer", "Sorry, I could not generate an answer.")
        answer_text = answer_text.replace("\\n", "\n")

        sources = [
            Source(title=src.get("title", "N/A"), score=src.get("score", 0.0))
            for src in result.get("sources", [])
        ]

        return AnswerResponse(
            answer=answer_text,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")