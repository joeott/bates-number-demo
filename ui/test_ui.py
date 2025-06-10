#!/usr/bin/env python3
"""
Test script to verify UI functionality.
Run this after starting both API and frontend servers.
"""

import requests
import time
import json

def test_api_health():
    """Test API health endpoint."""
    print("1. Testing API Health...")
    response = requests.get("http://localhost:8003/health")
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Agent Loaded: {data['agent_loaded']}")
    assert data['status'] == 'healthy'
    assert data['agent_loaded'] == True
    print("   ✓ API is healthy\n")

def test_api_stats():
    """Test API stats endpoint."""
    print("2. Testing API Stats...")
    response = requests.get("http://localhost:8003/api/stats")
    data = response.json()
    print(f"   Total Requests: {data['total_requests']}")
    print(f"   Error Rate: {data['error_rate']:.2%}")
    print("   ✓ Stats endpoint working\n")

def test_query_submission():
    """Test submitting a query."""
    print("3. Testing Query Submission...")
    
    test_queries = [
        "What are the main allegations in this case?",
        "Who are the parties involved?",
        "What damages are being claimed?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8003/api/query",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: Success")
            print(f"   Processing Time: {elapsed:.2f}s")
            print(f"   Answer Preview: {data['answer'][:100]}...")
            print(f"   Request ID: {data['request_id']}")
        else:
            print(f"   Status: Failed ({response.status_code})")
            print(f"   Error: {response.text}")
        
        time.sleep(1)  # Small delay between queries

def test_frontend_access():
    """Test frontend is accessible."""
    print("\n4. Testing Frontend Access...")
    response = requests.get("http://localhost:8000")
    print(f"   Status Code: {response.status_code}")
    print(f"   Content Type: {response.headers.get('content-type')}")
    assert response.status_code == 200
    assert 'text/html' in response.headers.get('content-type', '')
    print("   ✓ Frontend is accessible\n")

def test_rate_limiting():
    """Test rate limiting."""
    print("5. Testing Rate Limiting...")
    print("   Making 12 rapid requests...")
    
    hit_limit = False
    for i in range(12):
        response = requests.post(
            "http://localhost:8003/api/query",
            json={"query": f"Test query {i}"}
        )
        if response.status_code == 429:
            print(f"   Rate limit hit at request {i+1}")
            print(f"   Retry-After: {response.headers.get('Retry-After')}s")
            hit_limit = True
            break
    
    if hit_limit:
        print("   ✓ Rate limiting is working")
    else:
        print("   ⚠ Rate limit not hit (may be disabled)")

def print_ui_access_info():
    """Print information about accessing the UI."""
    print("\n" + "="*60)
    print("UI ACCESS INFORMATION")
    print("="*60)
    print("\nThe Legal Document Retriever UI is now running!")
    print("\nAccess the UI at: http://localhost:8000")
    print("\nAPI Documentation: http://localhost:8003/api/docs")
    print("\nAPI Health Check: http://localhost:8003/health")
    print("\nAPI Statistics: http://localhost:8003/api/stats")
    print("\nTo stop the servers, run: pkill -f 'python.*api.py|http.server'")
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Legal Document Retriever UI Test Suite")
    print("=====================================\n")
    
    try:
        test_api_health()
        test_api_stats()
        test_frontend_access()
        test_query_submission()
        test_rate_limiting()
        
        print("\n✅ All tests completed!")
        print_ui_access_info()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure both servers are running:")
        print("1. API: python ui/backend/api.py")
        print("2. Frontend: python -m http.server 8000 --directory ui/frontend")