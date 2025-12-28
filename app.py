"""Main FastAPI application entrypoint for enterprise-mlops-platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Platform API",
    description="Enterprise MLOps lifecycle management platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "MLOps Platform API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.on_event("startup")
async def startup():
    logger.info("MLOps Platform API starting up...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
