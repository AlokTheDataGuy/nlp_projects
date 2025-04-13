# File: chatbot/app.py
# Gradio interface for AI chatbot with better organization and UX

import gradio as gr
import webbrowser
from datetime import datetime

from chatbot.llm import OllamaClient
from database.logger import (
    create_session,
    log_conversation,
    end_session,
    update_conversation_feedback
)

# Initialize LLM client and active session tracking
llm = OllamaClient()
active_sessions = {}

def handle_chat(user_input, chat_history, session_id):
    """Handle user input, generate bot response, and update history"""
    if not user_input.strip():
        return chat_history

    if session_id not in active_sessions:
        session_id = create_session()
        active_sessions[session_id] = {"start_time": datetime.now()}

    # Get response and analyze topic
    bot_response = llm.get_completion(
        prompt=user_input,
        system_prompt="You are a helpful AI assistant. Be concise and clear in your responses."
    )
    topic = llm.analyze_topic(user_input)

    # Log conversation to database
    try:
        conv_id = log_conversation(session_id, user_input, bot_response, topic)
        active_sessions[session_id].setdefault("conversations", []).append(conv_id)
    except Exception as e:
        print(f"Error logging conversation: {e}")
        # Continue even if logging fails

    # Update chat history
    new_history = chat_history.copy() if chat_history else []
    new_history.append((user_input, bot_response))
    return new_history

def submit_feedback(session_id, conversation_index, rating):
    """Submit feedback for a conversation"""
    try:
        if session_id not in active_sessions:
            return f"Error: Session {session_id} not found"

        convo_ids = active_sessions[session_id].get("conversations", [])
        if not convo_ids or conversation_index >= len(convo_ids):
            return f"Error: Conversation index {conversation_index} not found"

        # Get the conversation ID and update the feedback
        conv_id = convo_ids[conversation_index]
        update_conversation_feedback(
            conversation_id=conv_id,
            satisfaction_rating=rating,
            query_success=(rating >= 3)  # Consider ratings of 3+ as successful
        )

        return f"✅ Feedback submitted for conversation #{conversation_index + 1}"
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return f"Error: {str(e)}"

def reset_session(session_id):
    """Ends the current session and resets session state"""
    if session_id in active_sessions:
        end_session(session_id)
        del active_sessions[session_id]
        return "✅ Session ended. New session will be created."
    return "❓ No session found."

def open_dashboard():
    """Opens the analytics dashboard in a new browser tab"""
    webbrowser.open("http://localhost:8501")
    return "📊 Analytics dashboard opened."

def clear_chat():
    """Clears the chat history"""
    return []

def launch_interface():
    """Gradio app interface definition and launch"""

    # Custom CSS for larger text
    custom_css = """
    body, .gradio-container, .gradio-container *, .prose, .prose *, .message, .message-wrap, .message-wrap * {
        font-size: 18px !important;
    }

    .chatbot-message {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    .chatbot .user-message, .chatbot .bot-message {
        font-size: 18px !important;
    }

    button, textarea, input, select, .gr-box, .gr-form, .gr-input, .gr-button {
        font-size: 18px !important;
    }

    h1, .gr-prose h1 {
        font-size: 36px !important;
        font-weight: bold !important;
    }

    h2, .gr-prose h2 {
        font-size: 28px !important;
        font-weight: bold !important;
    }

    h3, .gr-prose h3 {
        font-size: 22px !important;
        font-weight: bold !important;
    }

    .gr-button {
        min-height: 44px !important;
    }

    .gr-input, .gr-textarea {
        line-height: 1.5 !important;
        padding: 12px !important;
    }
    """

    with gr.Blocks(title="AI Chatbot", css=custom_css) as demo:
        gr.Markdown("# 🤖 AI Chatbot")
        gr.Markdown("Chat with our AI assistant powered by Cogito:8b")

        with gr.Accordion("📋 Instructions", open=True):
            gr.Markdown("""
            1. Enter a message and press Send
            2. All interactions are logged for analytics
            3. Use the feedback buttons after each response
            4. Access analytics dashboard for insights
            """)

        # State holders
        current_session_id = create_session()
        active_sessions[current_session_id] = {"start_time": datetime.now()}

        session_id = gr.State(current_session_id)
        chat_history = gr.State([])
        current_conv_index = gr.State(0)

        # Chat UI
        chatbot_display = gr.Chatbot(height=500)
        user_input = gr.Textbox(placeholder="Type your message here...", lines=2)

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear Chat")

        # Feedback section
        gr.Markdown("## 🌟 Feedback")

        with gr.Row():
            thumbs_up = gr.Button("👍 Helpful")
            neutral_btn = gr.Button("😐 Neutral")
            thumbs_down = gr.Button("👎 Not Helpful")

        feedback_status = gr.Textbox(label="Feedback Status", interactive=False)

        # Analytics section
        gr.Markdown("## 📊 Analytics")
        with gr.Row():
            end_button = gr.Button("End Session & Start New")
            dashboard_button = gr.Button("Open Analytics", variant="secondary")

        dashboard_output = gr.Textbox(label="Analytics Status", interactive=False)

        # Chat Interactions
        def process_chat(user_message, history, sess_id):
            """Process chat and update conversation index"""
            updated_history = handle_chat(user_message, history, sess_id)
            current_index = len(updated_history) - 1 if updated_history else 0
            return updated_history, updated_history, "", current_index

        send_button.click(
            process_chat,
            inputs=[user_input, chat_history, session_id],
            outputs=[chat_history, chatbot_display, user_input, current_conv_index]
        )

        user_input.submit(
            process_chat,
            inputs=[user_input, chat_history, session_id],
            outputs=[chat_history, chatbot_display, user_input, current_conv_index]
        )

        # Feedback button handlers
        thumbs_up.click(
            lambda idx, sess_id: submit_feedback(sess_id, idx, 5),
            inputs=[current_conv_index, session_id],
            outputs=[feedback_status]
        )

        neutral_btn.click(
            lambda idx, sess_id: submit_feedback(sess_id, idx, 3),
            inputs=[current_conv_index, session_id],
            outputs=[feedback_status]
        )

        thumbs_down.click(
            lambda idx, sess_id: submit_feedback(sess_id, idx, 1),
            inputs=[current_conv_index, session_id],
            outputs=[feedback_status]
        )

        # End session & reset
        def end_and_reset():
            """End the current session and create a new one"""
            new_session_id = create_session()
            active_sessions[new_session_id] = {"start_time": datetime.now()}
            return new_session_id, []

        end_button.click(
            reset_session,
            inputs=[session_id],
            outputs=[dashboard_output]
        ).then(
            end_and_reset,
            outputs=[session_id, chat_history]
        ).then(
            lambda history: history,
            inputs=[chat_history],
            outputs=[chatbot_display]
        )

        # Analytics
        dashboard_button.click(open_dashboard, outputs=[dashboard_output])

        # Clear chat
        clear_button.click(
            clear_chat,
            outputs=[chat_history]
        ).then(
            lambda history: history,
            inputs=[chat_history],
            outputs=[chatbot_display]
        )

    demo.launch(server_name="0.0.0.0", share=True)
    return demo
