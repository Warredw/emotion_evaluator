# Emotion Evaluator, Sentiment Analysis with VADER and BERT

This project is an NLP tool that classifies reviews as positive or negative using two pre-trained models: VADER and BERT. It includes benchmarking, an interactive Streamlit web app, and an optional FastAPI REST API.

---

##  Features

  - Classifies reviews using:
  - VADER (lexicon-based)
  - BERT (transformer-based, deep learning)
- Supports:
  - Single review input
  - CSV upload
  - API access via FastAPI
  -  Interactive Streamlit demo
  -  Benchmarking on IMDB movie review dataset


## Installation


git clone https://github.com/yourusername/emotion-evaluator.git
cd emotion_evaluator
python -m venv .venv
.venv\Scripts\activate   # On Windows
# or source .venv/bin/activate   # On macOS/Linux
pip install -r requirements.txt

##  What to Run and Why

### 1. **Launch the Streamlit Web App**
streamlit run app.py

### 2.  **Run the Benchmark Script**
python main.py

### 3.  **Start the FastAPI REST API**
uvicorn api_app:app --reload




