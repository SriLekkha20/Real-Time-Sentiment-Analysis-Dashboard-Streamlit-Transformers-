# Real-Time Sentiment Analysis Dashboard 💬

A simple interactive dashboard that simulates **live text streams** (e.g., tweets, reviews)
and visualizes their sentiment over time using a pre-trained transformer model.

## Features

- Uses a Hugging Face sentiment analysis pipeline
- Interactive text input for streaming-like behavior
- CSV import for existing social or feedback exports
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
streamlit run app.py
```

## CSV Import

Use the sidebar uploader to import message history from a CSV file. The app
detects common text columns from Xquik exports and other datasets:

- `text`
- `tweet_text`
- `full_text`
- `message`
- `content`
- `comment`
- `comments`
- `feedback`
- `review`
- `reviews`

If a `timestamp` column exists, the dashboard preserves it for the score trend.
