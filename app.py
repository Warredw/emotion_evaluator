import streamlit as st
import pandas as pd
from models.vader import VaderModel
from models.bert import BertModel
from evaluate import evaluate_csv

st.set_page_config(page_title="Emotion Evaluator", layout="wide")

vader_model = VaderModel()
bert_model = BertModel()

st.title("📊 Sentiment Evaluator — VADER vs BERT")

tabs = st.tabs(["📝 Single Review", "📁 Upload CSV"])


with tabs[0]:
    user_input = st.text_area("Enter a review:")
    if st.button("Analyze"):
        if user_input.strip():
            vader_score = vader_model.get_sentiment([user_input])[0]
            bert_score = bert_model.get_sentiment([user_input])[0]

            st.write("### Results")
            st.write({
                "VADER": {
                    "score": round(vader_score['compound'], 3),
                    "sentiment": "positive" if vader_score['compound'] >= 0 else "negative"
                },
                "BERT": {
                    "score": round(bert_score, 3),
                    "sentiment": "positive" if bert_score >= 0.5 else "negative"
                }
            })
        else:
            st.warning("Please enter a review before analyzing.")

with tabs[1]:
    uploaded_file = st.file_uploader("Upload a CSV with a 'review' column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("Uploaded CSV must contain a 'review' column.")
        else:
            df = evaluate_csv(df, vader_model, bert_model)

            st.write("### Preview of Predictions")
            st.dataframe(df[["review", "vader_score", "bert_score", "vader_sentiment", "bert_sentiment"]])

            # Offer download
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV with Predictions", csv_output, "predicted.csv", "text/csv")
