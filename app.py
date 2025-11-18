"""
Real-Time Sentiment Analysis Dashboard using Streamlit + Transformers.
"""

from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_sentiment_model():
    # Default sentiment-analysis pipeline (usually DistilBERT base finetuned model)
    return pipeline("sentiment-analysis")


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
            result = sentiment_analyzer(content)[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["records"].append(
                {
                    "timestamp": timestamp,
                    "text": content,
                    "label": result["label"],
                    "score": result["score"],
                }
            )
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
