# Emotion Evaluator, Sentiment Analysis with VADER and BERT

This project is an NLP tool that classifies IMDB movie reviews as positive or negative using two pre-trained models: VADER and BERT. It includes benchmarking, an interactive Streamlit web app, and an optional FastAPI REST API.

---

## 🚀 Features

- ✅ Classifies reviews using:
  - VADER (rule-based, lightweight)
  - BERT (transformer-based, deep learning)
- ✅ Supports:
  - Single review input
  - Batch CSV upload
  - API access via FastAPI
- ✅ Interactive Streamlit demo
- ✅ Benchmarking on IMDB movie review dataset
- ✅ Fully reproducible environment

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/emotion-evaluator.git
cd emotion-evaluator
python -m venv .venv
.venv\Scripts\activate   # On Windows
# or source .venv/bin/activate   # On macOS/Linux
pip install -r requirements.txt
