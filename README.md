# Real-Time Sentiment Analysis Dashboard 💬

A simple interactive dashboard that simulates **live text streams** (e.g., tweets, reviews)
and visualizes their sentiment over time using a pre-trained transformer model.

## Features

- Uses a Hugging Face sentiment analysis pipeline
- Interactive text input for streaming-like behavior
- Stores history of predictions during the session
- Visualizes:
  - Sentiment label counts
  - Sentiment score over time

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Pandas

## Installation

```bash
git clone https://github.com/<your-username>/sentiment-stream-dashboard.git
cd sentiment-stream-dashboard
pip install -r requirements.txt
