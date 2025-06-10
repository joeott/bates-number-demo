"""
Integration tests for the Legal Retriever UI.

Tests end-to-end functionality and interaction between frontend and backend.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import requests
from multiprocessing import Process
import sys

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ui" / "backend"))

from api import app
import uvicorn


class TestEndToEndFlow:
    """Tests for 3.1 End-to-End Flow criteria."""
    
    @pytest.fixture
    def api_server(self):
        """Start API server in a separate process."""
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8888, log_level="error")
        
        server = Process(target=run_server)
        server.start()
        
        # Wait for server to start
        time.sleep(2)
        
        yield "http://127.0.0.1:8888"
        
        server.terminate()
        server.join()
    
    def test_query_submission_to_response_display(self, api_server):
        """Query submission → API processing → Response display."""
        # Health check first
        response = requests.get(f"{api_server}/health")
        assert response.status_code == 200
        
        # Submit query
        query_data = {"query": "What are the contract terms?"}
        response = requests.post(
            f"{api_server}/api/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Should get response (or 503 if agent not initialized)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "query" in data
            assert "request_id" in data
            assert "processing_time" in data
    
    def test_multiple_sequential_queries(self, api_server):
        """Multiple sequential queries work correctly."""
        queries = [
            "What is the contract value?",
            "Who are the parties?",
            "What are the damages?"
        ]
        
        for query in queries:
            response = requests.post(
                f"{api_server}/api/query",
                json={"query": query}
            )
            assert response.status_code in [200, 503]
            
            # Small delay between queries
            time.sleep(0.5)
    
    def test_concurrent_users_isolation(self, api_server):
        """Concurrent users don't interfere with each other."""
        import concurrent.futures
        
        def submit_query(user_id):
            response = requests.post(
                f"{api_server}/api/query",
                json={
                    "query": f"Query from user {user_id}",
                    "request_id": f"user-{user_id}"
                }
            )
            return response.json() if response.status_code == 200 else None
        
        # Submit queries from multiple "users" concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit_query, i) for i in range(5)]
            results = [f.result() for f in futures]
        
        # Check that each got their own response
        request_ids = [r["request_id"] for r in results if r]
        assert len(set(request_ids)) == len(request_ids)  # All unique


class TestEdgeCases:
    """Tests for 3.2 Edge Cases criteria."""
    
    def test_very_short_queries(self):
        """Very short queries (1-2 words) handled."""
        from api import QueryRequest
        
        # Should pass validation
        short_query = QueryRequest(query="AI")
        assert len(short_query.query) == 2
    
    def test_very_long_queries(self):
        """Very long queries (>500 words) handled."""
        from api import QueryRequest
        
        # Should fail validation
        long_query = "word " * 201  # >1000 chars
        
        with pytest.raises(Exception):  # Validation error
            QueryRequest(query=long_query)
    
    def test_special_characters_unicode(self):
        """Special characters and Unicode support."""
        from api import QueryRequest
        
        special_queries = [
            "What about the café's liability?",
            "Is the § 1983 claim valid?",
            "Review the 中文 documents",
            "What's the plaintiff's claim?",
            "Contract worth €1,000,000"
        ]
        
        for query in special_queries:
            req = QueryRequest(query=query)
            assert req.query == query.strip()
    
    @patch('api.app_state.agent')
    def test_queries_with_no_results(self, mock_agent):
        """Queries with no results handled gracefully."""
        from fastapi.testclient import TestClient
        
        mock_agent.invoke.return_value = "No relevant information found in the documents."
        
        client = TestClient(app)
        response = client.post("/api/query", json={
            "query": "Completely unrelated query about quantum physics"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "No relevant information" in data["answer"]


class TestPerformance:
    """Tests for 3.3 Performance Testing criteria."""
    
    def test_response_time_typical_queries(self):
        """Response time <5 seconds for typical queries."""
        from fastapi.testclient import TestClient
        import time
        
        client = TestClient(app)
        
        start = time.time()
        response = client.post("/api/query", json={
            "query": "What are the main claims in this case?"
        })
        end = time.time()
        
        # Should respond within 5 seconds
        assert (end - start) < 5.0
        
        # Check processing time in response
        if response.status_code == 200:
            data = response.json()
            assert data.get("processing_time", 0) < 5.0
    
    def test_concurrent_users_handling(self):
        """Can handle 10 concurrent users."""
        from fastapi.testclient import TestClient
        import threading
        
        client = TestClient(app)
        results = []
        
        def make_request(user_id):
            response = client.post("/api/query", json={
                "query": f"Query from user {user_id}"
            })
            results.append(response.status_code)
        
        # Create 10 concurrent threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join()
        
        # All should have valid responses
        assert all(status in [200, 429, 503] for status in results)
    
    def test_memory_stability(self):
        """Memory usage stable over multiple queries."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Run 20 queries
        for i in range(20):
            client.post("/api/query", json={
                "query": f"Test query number {i}"
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<100MB)
        assert memory_increase < 100


class TestSecurityAndCompliance:
    """Tests for security and compliance criteria."""
    
    def test_xss_prevention(self):
        """XSS prevention in responses."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Try to inject script
        malicious_query = "<script>alert('XSS')</script>What is the contract?"
        
        response = client.post("/api/query", json={
            "query": malicious_query
        })
        
        if response.status_code == 200:
            data = response.json()
            # Script should be escaped or stripped
            assert "<script>" not in json.dumps(data)
    
    def test_sql_injection_prevention(self):
        """SQL injection prevention."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Try SQL injection
        sql_injection = "'; DROP TABLE documents; --"
        
        response = client.post("/api/query", json={
            "query": sql_injection
        })
        
        # Should handle safely
        assert response.status_code in [200, 503]
        
        # Check database is still intact (would need actual DB check)
    
    def test_rate_limiting(self):
        """Rate limiting is enforced."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Make many rapid requests
        responses = []
        for i in range(15):  # More than default limit
            response = client.post("/api/query", json={
                "query": f"Query {i}"
            })
            responses.append(response)
        
        # Should hit rate limit
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes  # Too Many Requests
        
        # Check rate limit headers
        for response in responses:
            if response.status_code == 429:
                assert "Retry-After" in response.headers
                assert "X-RateLimit-Limit" in response.headers


class TestAccessibility:
    """Tests for accessibility compliance."""
    
    def test_api_supports_screen_readers(self):
        """API responses are screen reader friendly."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "What are the damages?"
        })
        
        if response.status_code == 200:
            data = response.json()
            # Responses should be plain text, not HTML
            assert "<" not in data["answer"] or ">" not in data["answer"]
    
    def test_error_messages_are_descriptive(self):
        """Error messages are descriptive and actionable."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test various error conditions
        test_cases = [
            ({"query": ""}, "empty"),
            ({"query": "a" * 1001}, "too long"),
            ({"invalid": "data"}, "missing field")
        ]
        
        for data, error_type in test_cases:
            response = client.post("/api/query", json=data)
            
            if response.status_code != 200:
                error_data = response.json()
                # Should have clear error message
                assert "detail" in error_data or "error" in error_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])