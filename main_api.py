# main_api.py
import uvicorn
from db.core.database import init_db
from db.models import user, entity, query, query_llm_metric, query_timing
# Import the FastAPI app from your API package
from api.rag_api import app

if __name__ == "__main__":
    init_db()
    uvicorn.run(
        "main_api:app",   # Points to this file and the app object
        host="0.0.0.0",
        port=8000,
        reload=True       # Useful in dev, remove in production
    )