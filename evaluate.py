import pandas as pd

def evaluate_csv(df, vader_model, bert_model):
    
    # Clean up text
    df["review"] = df["review"].str.replace("<br />", " ", regex=False)

    # Get raw scores
    vader_results = vader_model.get_sentiment(df["review"].tolist())
    bert_results = bert_model.get_sentiment(df["review"].tolist())

    # Add compound score from VADER
    df["vader_score"] = [r["compound"] for r in vader_results]

    # Add BERT scores directly
    df["bert_score"] = bert_results

    # Create sentiment labels
    df["vader_sentiment"] = df["vader_score"].apply(lambda s: "positive" if s >= 0 else "negative")
    df["bert_sentiment"] = df["bert_score"].apply(lambda s: "positive" if s >= 0.5 else "negative")

    return df
