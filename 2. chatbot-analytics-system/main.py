# File: main.py
# This is the main file that integrates all components and starts the application

import os
import argparse
from datetime import datetime
import threading
import subprocess
import sys
import time
import webbrowser

def main():
    parser = argparse.ArgumentParser(description='Start Chatbot Interaction Analytics System')
    parser.add_argument('--mode', choices=['chatbot', 'dashboard', 'all'], default='all',
                      help='Mode to run: chatbot, dashboard, or all (default)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🤖 Chatbot Interaction Analytics System 📊")
    print("="*80)
    
    # Ensure the database exists
    from database.setup import setup_database
    setup_database()
    
    chatbot_url = "http://localhost:7860"
    dashboard_url = "http://localhost:8501"
    
    if args.mode in ['chatbot', 'all']:
        print(f"\n📋 Starting Chatbot Interface...")
        
        # Start chatbot in a separate thread if running both
        if args.mode == 'all':
            chatbot_thread = threading.Thread(target=start_chatbot)
            chatbot_thread.daemon = True
            chatbot_thread.start()
            print(f"✅ Chatbot started! Available at: {chatbot_url}")
            
            # Automatically open the chatbot in the default browser
            time.sleep(2)  # Give it a moment to start
            webbrowser.open(chatbot_url)
        else:
            print("⏳ Launching chatbot only mode...")
            start_chatbot()
            
    if args.mode in ['dashboard', 'all']:
        if args.mode == 'all':
            print(f"\n📊 Starting Analytics Dashboard...")
            print(f"✅ Dashboard will be available at: {dashboard_url}")
            print("\n💡 Recommended workflow:")
            print("   1. Use the chatbot first to generate interaction data")
            print("   2. Then explore the analytics dashboard to see insights")
            print("\n⏳ Starting services...")
        else:
            print("⏳ Launching dashboard only mode...")
            
        start_dashboard()

def start_chatbot():
    from chatbot.app import launch_interface
    try:
        launch_interface()
    except Exception as e:
        print(f"Error launching chatbot: {e}")
        print("Make sure Gradio is installed and Ollama is running with the Cogito model.")

def start_dashboard():
    # Using subprocess to run Streamlit as a separate process
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'app.py')
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Make sure Streamlit is installed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Application terminated by user. Thank you for using the Chatbot Analytics System!")