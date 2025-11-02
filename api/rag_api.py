"""
API Layer for Confluence RAG system.
Exposes endpoints for asking questions against the Confluence index.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from indexer.qdrant_utils import get_qdrant_client

# Import your modules
from config.logging_config import logger
from config.settings import LLM_MODEL_GENERATION, LLM_MODEL_REFINE, \
    LLM_BACKEND_TYPE_REFINEMENT, DEFAULT_TOP_K, RERANK_TOP_K, SOURCE_THRESHOLD, \
    API_ALLOWED_ORIGINS, REDIS_HOST, REDIS_PORT, REDIS_CACHE_TTL_DAYS, QDRANT_URL
from indexer.hybrid_index import HybridSearchIndex
from llm.bridge import LocalLLMBridge
from search.advanced_search import AdvancedSearch
from indexer.qdrant_utils import get_qdrant_stats
from llm.config import LLMConfig

# ---------------------------
# Request / Response Schemas
# ---------------------------

class QuestionRequest(BaseModel):
    question: str
    model: str
    search_top_k: Optional[int] = DEFAULT_TOP_K
    search_min_score: Optional[float] = SOURCE_THRESHOLD
    llm_top_k: Optional[int] = RERANK_TOP_K
    cache: Optional[bool] = True

class SearchRequest(BaseModel):
    question: str
    max_sources: Optional[int] = DEFAULT_TOP_K
    max_result: Optional[int] = RERANK_TOP_K

class Source(BaseModel):
    title: str
    score: float
    link: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
    services: dict


# ---------------------------
# App Initialization
# ---------------------------

app = FastAPI(title="Confluence RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,      # allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],        # allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],        # allows all headers
)

logger.info("Initializing Confluence RAG API...")

try:
    qdrant = get_qdrant_client(QDRANT_URL)
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

@app.get("/", tags=["General"])
def root():
    """Root endpoint"""
    return {
        "message": "Confluence RAG API",
        "version": "1.1.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint for Docker and Kubernetes"""
    services_status = {}
    overall_status = "healthy"

    # Check Qdrant
    try:
        _ = qdrant.get_collections()
        services_status["qdrant"] = "healthy"
    except Exception as ex:
        services_status["qdrant"] = f"unhealthy: {str(ex)}"
        overall_status = "unhealthy"

    # Check Hybrid Search Index
    try:
        if hybrid_search_index.is_fitted:
            services_status["hybrid_search"] = "healthy"
        else:
            services_status["hybrid_search"] = "not_fitted"
            overall_status = "degraded"
    except Exception as ex:
        services_status["hybrid_search"] = f"unhealthy: {str(ex)}"
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        services=services_status
    )


@app.post("/v1/rag/ask", response_model=AnswerResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    model_select = request.model if request.model is not None else LLM_MODEL_GENERATION

    if model_select not in LLMConfig.AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not available")

    model_params = LLMConfig.AVAILABLE_MODELS[model_select]
    model_backend = model_params.get("model_backend")

    """Ask a question against the indexed Confluence documentation"""
    rag_system_llm = LocalLLMBridge(
        search_system=search_system,
        generation_model_key=model_select,
        refinement_model_key=LLM_MODEL_REFINE,
        generation_model_backend_type=model_backend,
        refinement_model_backend_type=LLM_BACKEND_TYPE_REFINEMENT,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_cache_ttl_days=REDIS_CACHE_TTL_DAYS,
        enable_cache=request.cache,
    )
    setup_response = rag_system_llm.setup_model()

    if not setup_response:
        raise HTTPException(status_code=500, detail="LLM could not be initialized")

    try:
        result = rag_system_llm.ask(request.question,
                                top_k=request.search_top_k,
                                final_top_k=request.llm_top_k,
                                score_threshold=request.search_min_score)

        # Clean the answer formatting
        answer_text = result.get("answer", "Sorry, I could not generate an answer.")
        answer_text = answer_text.replace("\\n", "\n").replace("\\n", "\n")

        sources = [
            Source(title=src.get("title", "N/A"), score=src.get("score", 0.0), link=src.get("link", "N/A"))
            for src in result.get("sources", [])
        ]

        return AnswerResponse(
            answer=answer_text,
            sources=sources,
        )

    except Exception as ex:
        logger.error(f"Error during RAG query: {ex}")
        raise HTTPException(status_code=500, detail="Error processing question")

@app.get("/v1/rag/models", tags=["RAG"])
def get_available_models():
    return LLMConfig.AVAILABLE_MODELS

@app.post("/v1/rag/search", tags=["RAG"])
def semantic_search(request: SearchRequest):
    query = [request.question]
    max_sources = request.max_sources
    max_result = request.max_result

    if max_result > 5:
        return HTTPException(status_code=400, detail="The maximum number of results cannot exceed 5")
    if max_sources > 20:
        return HTTPException(status_code=400, detail="The maximum number of sources to sort cannot exceed 20")

    results = search_system.hybrid_search(query)
    out = []
    for res in results:
        out.append({
            "title": res.title,
            "score": res.score,
            "link": res.link,
            "last_updated": res.last_updated
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

@app.get("/v1/rag/stats", tags=["VectorDB"])
def rag_stats():
    return get_qdrant_stats(qdrant)