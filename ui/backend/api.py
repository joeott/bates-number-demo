"""
Legal Retriever API - FastAPI backend for the retrieval agent UI.

This module provides a REST API interface to the IterativeRetrieverAgent,
with proper error handling, validation, and monitoring capabilities.
"""

import logging
import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Import middleware
from middleware import RateLimitMiddleware, APIKeyAuth, SecurityHeadersMiddleware

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.retrieval_agent.main_retriever import IterativeRetrieverAgent
from src.retrieval_agent import agent_config
from src.utils import setup_logging
from src.config import VECTOR_STORE_PATH  # Load environment variables and paths

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    """Model for query requests."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="The legal question to search for"
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Maximum search iterations (1-5)"
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional request ID for tracking"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not empty after stripping."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Query cannot be empty')
        return cleaned

class QueryResponse(BaseModel):
    """Model for query responses."""
    answer: str
    query: str
    request_id: str
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Model for health check responses."""
    status: str
    agent_loaded: bool
    uptime_seconds: float
    version: str
    environment: str

# --- Application Setup ---
app = FastAPI(
    title="Legal Document Retriever API",
    description="AI-powered legal document search and analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
]

# Add production origins from environment
if os.getenv("ALLOWED_ORIGINS"):
    additional_origins = os.getenv("ALLOWED_ORIGINS").split(",")
    origins.extend(additional_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Add security headers
app.add_middleware(SecurityHeadersMiddleware)

# Add rate limiting
rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
rate_limit_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))
app.add_middleware(
    RateLimitMiddleware,
    calls_per_minute=rate_limit_per_minute,
    calls_per_hour=rate_limit_per_hour
)

# Optional API key auth
api_key_auth = APIKeyAuth(auto_error=False)

# --- Global State ---
class AppState:
    """Application state container."""
    def __init__(self):
        self.agent: Optional[IterativeRetrieverAgent] = None
        self.start_time: datetime = datetime.now()
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_processing_time: float = 0.0

app_state = AppState()

# --- Middleware ---
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(f"Request {request_id} completed in {process_time:.2f}s")
    
    return response

# --- Exception Handlers ---
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    app_state.error_count += 1
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An internal error occurred. Please try again later.",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# --- Startup/Shutdown ---
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting Legal Retriever API...")
    
    # Setup LangSmith if configured
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "legal-retriever-api")
        logger.info(f"LangSmith tracing enabled for project: {project}")
    
    # Initialize agent
    try:
        logger.info("Initializing IterativeRetrieverAgent...")
        app_state.agent = IterativeRetrieverAgent(
            max_iterations=agent_config.MAX_ITERATIONS
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        # Don't exit - allow health endpoint to report unhealthy state
        app_state.agent = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Legal Retriever API...")
    # Add any cleanup code here

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    uptime = (datetime.now() - app_state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if app_state.agent else "degraded",
        agent_loaded=app_state.agent is not None,
        uptime_seconds=uptime,
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Get API usage statistics."""
    uptime = (datetime.now() - app_state.start_time).total_seconds()
    avg_time = (
        app_state.total_processing_time / app_state.request_count 
        if app_state.request_count > 0 else 0
    )
    
    return {
        "uptime_seconds": uptime,
        "total_requests": app_state.request_count,
        "total_errors": app_state.error_count,
        "average_processing_time": avg_time,
        "error_rate": (
            app_state.error_count / app_state.request_count 
            if app_state.request_count > 0 else 0
        )
    }

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def process_query(
    request: QueryRequest, 
    req: Request,
    api_key: Optional[str] = Depends(api_key_auth)
):
    """Process a legal document query."""
    start_time = time.time()
    request_id = request.request_id or getattr(req.state, "request_id", str(uuid.uuid4()))
    
    # Update stats
    app_state.request_count += 1
    
    # Check agent availability
    if not app_state.agent:
        app_state.error_count += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval agent is not available. Please try again later."
        )
    
    try:
        logger.info(f"Processing query {request_id}: '{request.query[:50]}...'")
        
        # Prepare run configuration for LangSmith
        run_config = None
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            run_config = {
                "metadata": {
                    "request_id": request_id,
                    "api_request": True,
                    "max_iterations": request.max_iterations or agent_config.MAX_ITERATIONS,
                },
                "tags": ["api", "web_ui"],
                "run_name": f"API-{request.query[:30]}",
            }
        
        # Run the agent with timeout
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    app_state.agent.invoke,
                    request.query,
                    run_config=run_config
                ),
                timeout=120.0  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            app_state.error_count += 1
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Query processing timed out. Please try a simpler query."
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        app_state.total_processing_time += processing_time
        
        logger.info(f"Query {request_id} completed in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            request_id=request_id,
            processing_time=processing_time,
            success=True,
            metadata={
                "iterations_used": getattr(app_state.agent, "last_iteration_count", None),
                "facts_found": getattr(app_state.agent, "last_facts_count", None),
            }
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Error processing query {request_id}: {e}", exc_info=True)
        
        processing_time = time.time() - start_time
        app_state.total_processing_time += processing_time
        
        return QueryResponse(
            answer="",
            query=request.query,
            request_id=request_id,
            processing_time=processing_time,
            success=False,
            error=f"Failed to process query: {str(e)}"
        )

# --- Main ---
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Auto-reload: {reload}")
    logger.info(f"Documentation available at http://localhost:{port}/api/docs")
    
    # Run server
    uvicorn.run(
        "api:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )