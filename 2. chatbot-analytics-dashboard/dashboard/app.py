# File: dashboard/app.py
# Streamlit dashboard with default data for empty database

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
import sys

def get_db_path():
    # Get the absolute path to the database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, 'database', 'chatbot_analytics.db')

def create_sample_data():
    """Create sample data for demonstration when no real data exists"""
    # Sample conversation data
    conversations = {
        'id': list(range(1, 11)),
        'session_id': ['sample-session-1'] * 5 + ['sample-session-2'] * 5,
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
        'user_query': [
            'What can you help me with?',
            'How do I reset my password?',
            'Tell me about your features',
            'Can you explain how to use this app?',
            'I need technical support',
            'What time is it?',
            'How does the analytics work?',
            'I\'m having trouble with login',
            'Show me some examples',
            'Thank you for your help'
        ],
        'bot_response': ['Sample response ' + str(i) for i in range(1, 11)],
        'topic': [
            'General Inquiry',
            'Technical Support',
            'Product Information',
            'General Inquiry',
            'Technical Support',
            'General Inquiry',
            'Product Information',
            'Technical Support',
            'Product Information',
            'Feedback'
        ],
        'satisfaction_rating': [5, 4, 5, 3, 4, 5, 4, 3, 5, 5],
        'query_success': [True, True, True, False, True, True, True, False, True, True],
        'start_time': [datetime.now() - timedelta(hours=i, minutes=10) for i in range(10)],
        'end_time': [datetime.now() - timedelta(hours=i) for i in range(10)],
        'duration_seconds': [600, 500, 550, 480, 620, 580, 540, 490, 610, 520]
    }

    # Sample session data
    sessions = {
        'session_id': ['sample-session-1', 'sample-session-2'],
        'start_time': [datetime.now() - timedelta(hours=8), datetime.now() - timedelta(hours=4)],
        'end_time': [datetime.now() - timedelta(hours=6), datetime.now() - timedelta(hours=2)],
        'duration_seconds': [7200, 7800]
    }

    return pd.DataFrame(conversations), pd.DataFrame(sessions)

def load_conversation_data():
    """Load conversation data from the database or use sample data if empty"""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        st.warning("Database not found. Using sample data for demonstration.")
        sample_conv, _ = create_sample_data()
        return sample_conv

    conn = sqlite3.connect(db_path)
    query = """
    SELECT c.*, s.start_time, s.end_time, s.duration_seconds
    FROM conversations c
    JOIN sessions s ON c.session_id = s.session_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # If no data in database, use sample data
    if df.empty:
        st.warning("No conversation data found in database. Using sample data for demonstration.")
        sample_conv, _ = create_sample_data()
        return sample_conv

    # Convert timestamp strings to datetime objects
    for col in ['timestamp', 'start_time', 'end_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df

def load_session_data():
    """Load session data from the database or use sample data if empty"""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        _, sample_sess = create_sample_data()
        return sample_sess

    conn = sqlite3.connect(db_path)
    query = """
    SELECT * FROM sessions
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # If no data in database, use sample data
    if df.empty:
        _, sample_sess = create_sample_data()
        return sample_sess

    # Convert timestamp strings to datetime objects
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df

def generate_metrics(df_conv, df_sessions):
    """Generate metrics from conversation and session data"""
    metrics = {}

    # Total number of queries
    metrics['total_queries'] = len(df_conv)

    # Query success rate
    if not df_conv.empty and 'query_success' in df_conv.columns:
        success_data = df_conv['query_success'].dropna()
        if not success_data.empty:
            metrics['success_rate'] = round(100 * success_data.mean(), 1)
        else:
            metrics['success_rate'] = "No data"
    else:
        metrics['success_rate'] = "No data"

    # Average satisfaction rating
    if not df_conv.empty and 'satisfaction_rating' in df_conv.columns:
        rating_data = df_conv['satisfaction_rating'].dropna()
        if not rating_data.empty:
            metrics['avg_satisfaction'] = round(rating_data.mean(), 2)
        else:
            metrics['avg_satisfaction'] = "No data"
    else:
        metrics['avg_satisfaction'] = "No data"

    # Average session length
    if not df_sessions.empty and 'duration_seconds' in df_sessions.columns:
        duration_data = df_sessions['duration_seconds'].dropna()
        if not duration_data.empty:
            metrics['avg_session_length'] = round(duration_data.mean() / 60, 2)  # in minutes
        else:
            metrics['avg_session_length'] = "No data"
    else:
        metrics['avg_session_length'] = "No data"

    return metrics

def generate_topic_distribution(df):
    """Generate topic distribution visualization"""
    if df.empty or 'topic' not in df.columns:
        return go.Figure()

    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']

    fig = px.pie(
        topic_counts,
        values='Count',
        names='Topic',
        title='Conversation Topics Distribution',
        hole=0.4
    )
    return fig

def generate_usage_over_time(df):
    """Generate usage over time visualization"""
    if df.empty or 'timestamp' not in df.columns:
        return go.Figure()

    # Resample by day
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size().reset_index()
    daily_counts.columns = ['Date', 'Count']

    fig = px.line(
        daily_counts,
        x='Date',
        y='Count',
        title='Daily Usage Over Time',
        labels={'Count': 'Number of Queries', 'Date': 'Date'}
    )
    return fig

def generate_satisfaction_over_time(df):
    """Generate satisfaction over time visualization"""
    if df.empty or 'timestamp' not in df.columns or 'satisfaction_rating' not in df.columns:
        return go.Figure()

    # Filter out rows with no satisfaction rating
    df_rated = df.dropna(subset=['satisfaction_rating']).copy()  # Use .copy() to avoid SettingWithCopyWarning
    if df_rated.empty:
        return go.Figure()

    # Resample by day
    df_rated.loc[:, 'date'] = df_rated['timestamp'].dt.date  # Use .loc to avoid SettingWithCopyWarning
    daily_satisfaction = df_rated.groupby('date')['satisfaction_rating'].mean().reset_index()
    daily_satisfaction.columns = ['Date', 'Average Rating']

    fig = px.line(
        daily_satisfaction,
        x='Date',
        y='Average Rating',
        title='Average Satisfaction Rating Over Time',
        labels={'Average Rating': 'Average Rating (1-5)', 'Date': 'Date'}
    )

    # Add reference line at rating = 3 (neutral)
    fig.add_shape(
        type="line",
        x0=daily_satisfaction['Date'].min(),
        y0=3,
        x1=daily_satisfaction['Date'].max(),
        y1=3,
        line=dict(color="gray", width=1, dash="dash"),
    )

    return fig

def generate_hourly_usage(df):
    """Generate hourly usage pattern visualization"""
    if df.empty or 'timestamp' not in df.columns:
        return go.Figure()

    # Extract hour of day
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df.groupby('hour').size().reset_index()
    hourly_counts.columns = ['Hour', 'Count']

    # Ensure all hours are represented
    all_hours = pd.DataFrame({'Hour': range(24)})
    hourly_counts = pd.merge(all_hours, hourly_counts, on='Hour', how='left').fillna(0)

    fig = px.bar(
        hourly_counts,
        x='Hour',
        y='Count',
        title='Usage by Hour of Day',
        labels={'Count': 'Number of Queries', 'Hour': 'Hour of Day (24h)'}
    )
    return fig

def generate_top_queries(df, top_n=10):
    """Generate top user queries visualization"""
    if df.empty or 'user_query' not in df.columns:
        return go.Figure()

    # Get the most common user queries
    query_counts = df['user_query'].value_counts().head(top_n).reset_index()
    query_counts.columns = ['Query', 'Count']

    # Truncate long queries for display
    query_counts['Query'] = query_counts['Query'].apply(
        lambda x: x[:50] + '...' if len(x) > 50 else x
    )

    fig = px.bar(
        query_counts,
        y='Query',
        x='Count',
        title=f'Top {top_n} User Queries',
        labels={'Count': 'Frequency', 'Query': 'User Query'},
        orientation='h'
    )
    return fig

def main():
    """Main dashboard function"""
    st.set_page_config(page_title="Chatbot Analytics Dashboard", layout="wide")

    # Custom CSS for larger font sizes
    st.markdown("""
    <style>
        html, body, [class*="st-"] {
            font-size: 18px !important;
        }
        h1 {
            font-size: 2.5rem !important;
        }
        h2 {
            font-size: 2rem !important;
        }
        h3 {
            font-size: 1.5rem !important;
        }
        .stButton > button {
            font-size: 18px !important;
            padding: 0.5rem 1rem !important;
        }
        .stSelectbox > div > div, .stMultiSelect > div > div {
            font-size: 18px !important;
        }
        .stDataFrame, .stTable {
            font-size: 16px !important;
        }
        .stMarkdown p {
            font-size: 18px !important;
        }
        .stPlotlyChart, .stVegaLiteChart {
            font-size: 16px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Chatbot Interaction Analytics Dashboard")

    # Add introduction and instructions
    with st.expander("How to use this dashboard", expanded=True):
        st.write("""
        ## Getting Started

        1. **Use the chatbot first**: The chatbot interface is available at http://localhost:7860
        2. **Interact with the chatbot**: Ask questions, provide feedback on responses
        3. **View analytics**: This dashboard shows metrics from your interactions
        4. **Refresh data**: Click the refresh button to see updated analytics

        If you haven't used the chatbot yet, sample data will be shown for demonstration.
        """)

    # Chatbot link button
    if st.button("🤖 Open Chatbot Interface", type="primary"):
        st.markdown("[Chatbot opened in new tab](http://localhost:7860)", unsafe_allow_html=True)
        st.components.v1.html(
            """
            <script>
                window.open('http://localhost:7860', '_blank');
            </script>
            """,
            height=0
        )

    # Add refresh button
    if st.button("🔄 Refresh Analytics Data"):
        st.rerun()

    st.markdown("---")

    # Load data
    try:
        df_conv = load_conversation_data()
        df_sessions = load_session_data()

        # Generate metrics
        metrics = generate_metrics(df_conv, df_sessions)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", metrics['total_queries'])

        with col2:
            st.metric("Query Success Rate",
                     f"{metrics['success_rate']}%" if isinstance(metrics['success_rate'], (int, float)) else metrics['success_rate'])

        with col3:
            st.metric("Avg. Satisfaction", metrics['avg_satisfaction'])

        with col4:
            st.metric("Avg. Session Length",
                     f"{metrics['avg_session_length']} min" if isinstance(metrics['avg_session_length'], (int, float)) else metrics['avg_session_length'])

        # Main visualizations
        st.subheader("Conversation Analytics")

        # Row 1: Topic Distribution and Usage Over Time
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(generate_topic_distribution(df_conv), use_container_width=True)

        with col2:
            st.plotly_chart(generate_usage_over_time(df_conv), use_container_width=True)

        # Row 2: Satisfaction Over Time and Hourly Usage
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(generate_satisfaction_over_time(df_conv), use_container_width=True)

        with col2:
            st.plotly_chart(generate_hourly_usage(df_conv), use_container_width=True)

        # Top queries section
        st.subheader("Most Common User Queries")
        st.plotly_chart(generate_top_queries(df_conv), use_container_width=True)

        # Raw data exploration (optional)
        with st.expander("Explore Raw Data"):
            st.write("#### Sample of conversation data")
            st.dataframe(df_conv.head(10))

            if 'sample-session' not in df_conv['session_id'].iloc[0]:
                st.success("✅ Using real chatbot interaction data")
            else:
                st.warning("⚠️ Using sample data. Start interacting with the chatbot to generate real data.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please try refreshing the page or check that the database is properly set up.")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    # When called from main.py, this function doesn't actually
    # run the Streamlit app - that's handled by the subprocess
    pass

if __name__ == "__main__":
    main()