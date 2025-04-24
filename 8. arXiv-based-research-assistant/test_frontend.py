import requests
import sys
import webbrowser
import time

def test_frontend_server():
    """Test if the frontend server is running"""
    try:
        response = requests.get("http://localhost:5173")
        if response.status_code == 200:
            print("✅ Frontend server is running")
            return True
        else:
            print(f"❌ Frontend server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to frontend server")
        return False

def main():
    print("Testing frontend server...")
    
    # Test if frontend server is running
    frontend_ok = test_frontend_server()
    
    if frontend_ok:
        print("✅ Frontend server is running")
        
        # Open browser to test frontend
        print("Opening browser to test frontend...")
        webbrowser.open("http://localhost:5173")
        
        print("\nManual testing steps:")
        print("1. Check if the chat interface loads correctly")
        print("2. Try sending a message in the chat")
        print("3. Navigate to the Papers page and try searching for papers")
        print("4. Navigate to the Concepts page and check if it loads")
        
        print("\nPress Ctrl+C when done testing")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTesting complete")
    else:
        print("❌ Frontend server is not running")
        print("Make sure to start the frontend server with 'cd frontend && npm run dev'")
        sys.exit(1)

if __name__ == "__main__":
    main()
