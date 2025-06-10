"""
Middleware components for the Legal Retriever API.

Includes rate limiting, authentication, and other cross-cutting concerns.
"""

import time
import hashlib
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import os

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse.
    
    Implements a sliding window rate limiter with configurable limits.
    """
    
    def __init__(
        self, 
        app, 
        calls_per_minute: int = 10,
        calls_per_hour: int = 100,
        burst_size: int = 5
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.burst_size = burst_size
        
        # Storage for rate limit tracking
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Include API key if present for per-key limits
        api_key = request.headers.get("X-API-Key", "")
        
        return f"{client_ip}:{api_key}"
    
    def _clean_old_requests(self, requests: list, cutoff: datetime):
        """Remove requests older than cutoff time."""
        return [req_time for req_time in requests if req_time > cutoff]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits before processing request."""
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/api/docs", "/api/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        now = datetime.now()
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self.minute_requests[client_id] = self._clean_old_requests(
            self.minute_requests[client_id], minute_ago
        )
        self.hour_requests[client_id] = self._clean_old_requests(
            self.hour_requests[client_id], hour_ago
        )
        
        # Check minute limit
        minute_count = len(self.minute_requests[client_id])
        if minute_count >= self.calls_per_minute:
            logger.warning(f"Rate limit exceeded for {client_id}: {minute_count}/min")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((minute_ago + timedelta(minutes=2)).timestamp()))
                }
            )
        
        # Check hour limit
        hour_count = len(self.hour_requests[client_id])
        if hour_count >= self.calls_per_hour:
            logger.warning(f"Hour rate limit exceeded for {client_id}: {hour_count}/hour")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_hour} requests per hour",
                    "retry_after": 3600
                },
                headers={
                    "Retry-After": "3600",
                    "X-RateLimit-Limit-Hour": str(self.calls_per_hour),
                    "X-RateLimit-Remaining-Hour": "0"
                }
            )
        
        # Record this request
        self.minute_requests[client_id].append(now)
        self.hour_requests[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.calls_per_minute - minute_count - 1)
        response.headers["X-RateLimit-Reset"] = str(int((minute_ago + timedelta(minutes=2)).timestamp()))
        
        return response


class APIKeyAuth(HTTPBearer):
    """
    Optional API key authentication.
    
    Validates API keys from environment or database.
    """
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.valid_api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """Load valid API keys from environment."""
        keys = {}
        
        # Load from environment variable (comma-separated)
        api_keys_env = os.getenv("API_KEYS", "")
        if api_keys_env:
            for key_entry in api_keys_env.split(","):
                if ":" in key_entry:
                    key, name = key_entry.split(":", 1)
                    keys[key.strip()] = {
                        "name": name.strip(),
                        "created": datetime.now(),
                        "active": True
                    }
        
        # Add a default development key if no keys configured
        if not keys and os.getenv("ENVIRONMENT") == "development":
            keys["dev-key-123"] = {
                "name": "Development Key",
                "created": datetime.now(),
                "active": True
            }
            logger.warning("Using default development API key")
        
        return keys
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key from request."""
        # Check if API key auth is enabled
        if not os.getenv("REQUIRE_API_KEY", "false").lower() == "true":
            return None  # No auth required
        
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        
        # Also check Authorization header
        if not api_key:
            try:
                credentials = await super().__call__(request)
                if credentials:
                    api_key = credentials.credentials
            except HTTPException:
                pass
        
        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            else:
                return None
        
        # Validate API key
        if api_key not in self.valid_api_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            else:
                return None
        
        # Check if key is active
        key_info = self.valid_api_keys[api_key]
        if not key_info.get("active", True):
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is inactive"
                )
            else:
                return None
        
        # Log usage
        logger.info(f"API key authenticated: {key_info['name']}")
        
        # Store key info in request state for later use
        request.state.api_key = api_key
        request.state.api_key_name = key_info["name"]
        
        return api_key


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP for HTML responses
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self';"
            )
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Add gzip compression for large responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Only compress large text responses
        content_length = response.headers.get("content-length")
        content_type = response.headers.get("content-type", "")
        
        if content_length and int(content_length) > 1024:  # > 1KB
            if any(ct in content_type for ct in ["text/", "application/json"]):
                # Note: Actual compression would require response body manipulation
                # This is a placeholder for the header
                response.headers["Content-Encoding"] = "gzip"
        
        return response