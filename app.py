"""Main FastAPI application entrypoint for enterprise-mlops-platform."""
import os
import logging
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Restrict origins to known frontends; override via ALLOWED_ORIGINS env var
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080"
).split(",")

app = FastAPI(
    title="MLOps Platform API",
    description="Enterprise MLOps lifecycle management platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # FIX: credentials require specific origins, not wildcard
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "MLOps Platform API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
