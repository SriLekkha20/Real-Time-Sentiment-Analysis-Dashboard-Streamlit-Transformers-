"""
Real-Time Sentiment Analysis Dashboard using Streamlit + Transformers.
"""

from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

TEXT_COLUMN_PRIORITY = (
    "text",
    "tweet_text",
    "full_text",
    "message",
    "content",
    "comment",
    "comments",
    "feedback",
    "review",
    "reviews",
)


@st.cache_resource
def load_sentiment_model():
    # Default sentiment-analysis pipeline (usually DistilBERT base finetuned model)
    return pipeline("sentiment-analysis")


def find_text_column(columns):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_PRIORITY:
        if candidate in normalized:
            return normalized[candidate]
    return None


def analyze_message(content, sentiment_analyzer, timestamp=None):
    result = sentiment_analyzer(content)[0]
    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": content,
        "label": result["label"],
        "score": result["score"],
    }


def main():
    st.set_page_config(
        page_title="Sentiment Stream Dashboard",
        layout="wide",
    )

    st.title("💬 Real-Time Sentiment Analysis Dashboard")
    st.write(
        "Type messages below to simulate an incoming stream of social media posts, "
        "product reviews, or customer feedback."
    )

    sentiment_analyzer = load_sentiment_model()

    if "records" not in st.session_state:
        st.session_state["records"] = []

    with st.sidebar:
        st.header("Controls")
        uploaded_csv = st.file_uploader("Import message CSV", type=["csv"])
        if uploaded_csv:
            uploaded_data = pd.read_csv(uploaded_csv)
            text_column = find_text_column(uploaded_data.columns)
            if text_column is None:
                st.warning("No supported text column found in the CSV.")
            else:
                timestamp_column = "timestamp" if "timestamp" in uploaded_data.columns else None
                imported_count = 0
                for _, row in uploaded_data.iterrows():
                    content = str(row[text_column]).strip()
                    if not content or content.lower() == "nan":
                        continue
                    timestamp = row[timestamp_column] if timestamp_column else None
                    st.session_state["records"].append(
                        analyze_message(content, sentiment_analyzer, timestamp=timestamp)
                    )
                    imported_count += 1
                st.success(f"Imported {imported_count} messages.")

        reset = st.button("Clear History")
        if reset:
            st.session_state["records"] = []
            st.success("History cleared.")

    input_text = st.text_area(
        "Enter text (one message at a time):",
        placeholder="Example: I love this new AI-powered feature!",
        height=120,
    )

    if st.button("Analyze Sentiment"):
        content = input_text.strip()
        if content:
            st.session_state["records"].append(analyze_message(content, sentiment_analyzer))
        else:
            st.warning("Please enter some text before analyzing.")

    data = pd.DataFrame(st.session_state["records"])

    if not data.empty:
        st.subheader("Recent Messages")
        st.dataframe(
            data[["timestamp", "text", "label", "score"]].sort_values(
                by="timestamp", ascending=False
            ),
            use_container_width=True,
            height=300,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            st.bar_chart(data["label"].value_counts())

        with col2:
            st.subheader("Sentiment Score Over Time")
            line_data = data.copy()
            line_data["timestamp"] = pd.to_datetime(line_data["timestamp"])
            line_data = line_data.set_index("timestamp")[["score"]]
            st.line_chart(line_data)

    else:
        st.info("No messages analyzed yet. Start by entering text above.")


if __name__ == "__main__":
    main()
