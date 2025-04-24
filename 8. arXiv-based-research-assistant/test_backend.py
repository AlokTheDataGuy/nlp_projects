import requests
import json
import sys

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health endpoint is working")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Health endpoint returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        return False

def test_paper_search():
    """Test the paper search endpoint"""
    try:
        response = requests.post(
            "http://localhost:8000/api/papers/search",
            json={"query": "machine learning", "max_results": 3}
        )
        if response.status_code == 200:
            papers = response.json().get("papers", [])
            if papers:
                print(f"✅ Paper search endpoint is working, found {len(papers)} papers")
                print(f"First paper: {papers[0]['title']}")
                return True
            else:
                print("❌ Paper search endpoint returned no papers")
                return False
        else:
            print(f"❌ Paper search endpoint returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        return False

def test_chat():
    """Test the chat endpoint"""
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "What is machine learning?", "conversation_history": []}
        )
        if response.status_code == 200:
            print("✅ Chat endpoint is working")
            print(f"Response: {response.json().get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Chat endpoint returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        return False

def main():
    print("Testing backend API...")
    
    # Run tests
    health_ok = test_health()
    
    if not health_ok:
        print("❌ Backend server is not running or not healthy")
        sys.exit(1)
    
    paper_search_ok = test_paper_search()
    chat_ok = test_chat()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Health endpoint: {'✅' if health_ok else '❌'}")
    print(f"Paper search endpoint: {'✅' if paper_search_ok else '❌'}")
    print(f"Chat endpoint: {'✅' if chat_ok else '❌'}")
    
    if health_ok and paper_search_ok and chat_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
