# main_api.py
import uvicorn

# Import the FastAPI app from your API package
from api.rag_api import app

if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",   # Points to this file and the app object
        host="0.0.0.0",
        port=8000,
        reload=True       # Useful in dev, remove in production
    )