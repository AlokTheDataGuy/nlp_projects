import subprocess
import sys
import os
import time
import webbrowser
import signal
import atexit

# Define processes
backend_process = None
frontend_process = None

def cleanup():
    """Clean up processes on exit"""
    if backend_process:
        print("Stopping backend...")
        if sys.platform == 'win32':
            backend_process.kill()
        else:
            os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
    
    if frontend_process:
        print("Stopping frontend...")
        if sys.platform == 'win32':
            frontend_process.kill()
        else:
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)

# Register cleanup function
atexit.register(cleanup)

def main():
    global backend_process, frontend_process
    
    print("Starting ArXiv Expert Chatbot...")
    
    # Start backend
    print("Starting backend server...")
    os.chdir("backend")
    backend_process = subprocess.Popen(
        ["python", "run.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True if sys.platform == 'win32' else False,
        preexec_fn=None if sys.platform == 'win32' else os.setsid
    )
    os.chdir("..")
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(5)
    
    # Start frontend
    print("Starting frontend server...")
    os.chdir("frontend")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True if sys.platform == 'win32' else False,
        preexec_fn=None if sys.platform == 'win32' else os.setsid
    )
    os.chdir("..")
    
    # Wait for frontend to start
    print("Waiting for frontend to start...")
    time.sleep(5)
    
    # Open browser
    print("Opening browser...")
    webbrowser.open("http://localhost:5173")
    
    print("ArXiv Expert Chatbot is running!")
    print("Press Ctrl+C to stop the servers.")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")

if __name__ == "__main__":
    main()
