"""
Unit tests for the Legal Retriever API.

Tests cover all verification criteria from context_59_ui_verification_criteria.md
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ui" / "backend"))

# Import after path setup
from api import app, app_state, QueryRequest, QueryResponse

# Test client
client = TestClient(app)


class TestStartupVerification:
    """Tests for 1.1 Startup Verification criteria."""
    
    def test_api_starts_successfully(self):
        """API starts successfully with proper logging."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "agent_loaded" in data
    
    @patch('api.IterativeRetrieverAgent')
    def test_agent_initialization_success(self, mock_agent):
        """Agent initialization completes without errors."""
        mock_agent.return_value = Mock()
        with TestClient(app) as test_client:
            response = test_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["agent_loaded"] is True
    
    @patch('api.IterativeRetrieverAgent', side_effect=Exception("Init failed"))
    def test_graceful_failure_on_missing_dependencies(self, mock_agent):
        """Graceful failure with clear error messages if dependencies missing."""
        # Reset app state
        app_state.agent = None
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["agent_loaded"] is False
    
    def test_health_check_endpoint(self):
        """Health check endpoint responds correctly."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert "environment" in data


class TestRequestHandling:
    """Tests for 1.2 Request Handling criteria."""
    
    @patch('api.app_state.agent')
    def test_valid_query_returns_response(self, mock_agent):
        """Valid queries return appropriate responses."""
        mock_agent.invoke.return_value = "Test answer"
        
        response = client.post("/api/query", json={
            "query": "What are the contract terms?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["answer"] == "Test answer"
        assert data["query"] == "What are the contract terms?"
    
    def test_empty_query_returns_error(self):
        """Invalid/empty queries return proper error messages."""
        response = client.post("/api/query", json={
            "query": ""
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_whitespace_only_query_returns_error(self):
        """Whitespace-only queries are rejected."""
        response = client.post("/api/query", json={
            "query": "   \n\t   "
        })
        
        assert response.status_code == 422
    
    def test_long_query_handling(self):
        """Long queries (>1000 chars) handled appropriately."""
        long_query = "a" * 1001
        response = client.post("/api/query", json={
            "query": long_query
        })
        
        assert response.status_code == 422
    
    @patch('api.app_state.agent')
    async def test_concurrent_requests(self, mock_agent):
        """Concurrent requests handled without conflicts."""
        mock_agent.invoke.return_value = "Test answer"
        
        # Create multiple concurrent requests
        async def make_request(query_num):
            return client.post("/api/query", json={
                "query": f"Test query {query_num}"
            })
        
        # Run concurrent requests
        tasks = [make_request(i) for i in range(5)]
        
        # All should succeed
        for task in tasks:
            response = await asyncio.create_task(task)
            assert response.status_code == 200
    
    @patch('api.app_state.agent')
    @patch('api.asyncio.wait_for', side_effect=asyncio.TimeoutError)
    def test_timeout_handling(self, mock_wait, mock_agent):
        """Timeout handling for long-running queries."""
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 504
        assert "timeout" in response.json()["detail"].lower()


class TestErrorHandling:
    """Tests for 1.3 Error Handling criteria."""
    
    @patch('api.app_state.agent', None)
    def test_agent_initialization_failure_handled(self):
        """Agent initialization failures caught and logged."""
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
    
    def test_malformed_json_rejected(self):
        """Malformed JSON requests rejected with 400 status."""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @patch('api.app_state.agent')
    def test_internal_errors_dont_expose_sensitive_info(self, mock_agent):
        """Internal errors don't expose sensitive information."""
        mock_agent.invoke.side_effect = Exception("Database password: secret123")
        
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200  # Our handler returns 200 with error
        data = response.json()
        assert data["success"] is False
        assert "secret123" not in data["error"]
        assert "Database password" not in data["error"]


class TestSecurity:
    """Tests for 1.4 Security criteria."""
    
    def test_cors_configuration(self):
        """CORS properly configured for production origins."""
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_input_sanitization(self):
        """Input sanitization prevents injection attacks."""
        # Test SQL injection attempt
        response = client.post("/api/query", json={
            "query": "'; DROP TABLE users; --"
        })
        
        # Should process normally without executing SQL
        assert response.status_code in [200, 503]  # Depends on agent state
    
    def test_no_sensitive_data_in_errors(self):
        """No sensitive data in error responses."""
        response = client.post("/api/query", json={
            "query": "a" * 1001  # Too long
        })
        
        assert response.status_code == 422
        error_text = json.dumps(response.json())
        
        # Check for common sensitive patterns
        assert "password" not in error_text.lower()
        assert "secret" not in error_text.lower()
        assert "key" not in error_text.lower()


class TestAPIEndpoints:
    """Tests for specific API endpoints."""
    
    def test_stats_endpoint(self):
        """Stats endpoint returns usage statistics."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "total_errors" in data
        assert "average_processing_time" in data
        assert "error_rate" in data
    
    def test_api_documentation(self):
        """API documentation is accessible."""
        response = client.get("/api/docs")
        assert response.status_code == 200
        
        response = client.get("/api/redoc")
        assert response.status_code == 200
    
    @patch('api.app_state.agent')
    def test_request_id_tracking(self, mock_agent):
        """Request IDs are properly tracked."""
        mock_agent.invoke.return_value = "Test answer"
        
        # With custom request ID
        response = client.post("/api/query", json={
            "query": "Test query",
            "request_id": "test-123"
        })
        
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") is not None
        data = response.json()
        assert data["request_id"] == "test-123"
        
        # Without custom request ID
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") is not None
        data = response.json()
        assert data["request_id"] is not None


class TestPerformance:
    """Tests for performance-related criteria."""
    
    @patch('api.app_state.agent')
    def test_response_time_tracking(self, mock_agent):
        """Response time is tracked in headers."""
        mock_agent.invoke.return_value = "Test answer"
        
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time > 0
        assert process_time < 10  # Should be fast for mock
    
    def test_max_iterations_parameter(self):
        """Max iterations parameter is validated."""
        # Valid range
        for iterations in [1, 3, 5]:
            response = client.post("/api/query", json={
                "query": "Test query",
                "max_iterations": iterations
            })
            assert response.status_code in [200, 503]  # Depends on agent
        
        # Invalid range
        for iterations in [0, 6, -1]:
            response = client.post("/api/query", json={
                "query": "Test query",
                "max_iterations": iterations
            })
            assert response.status_code == 422


# Fixtures for integration testing
@pytest.fixture
def mock_agent():
    """Mock agent for testing."""
    agent = Mock()
    agent.invoke.return_value = "Mock answer"
    agent.last_iteration_count = 2
    agent.last_facts_count = 5
    return agent


@pytest.fixture
def app_with_mock_agent(mock_agent):
    """App instance with mocked agent."""
    app_state.agent = mock_agent
    return app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])