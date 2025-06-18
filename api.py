from fastapi import FastAPI, UploadFile
from models.bert import BertModel
from models.vader import VaderModel
from evaluate import evaluate_csv
import pandas as pd

app = FastAPI()
vader_model = VaderModel()
bert_model = BertModel()

@app.get("/")
def root():
    return {"message": "Welcome to the Emotion Evaluator API"}

@app.post("/predict/text")
def predict_text(review: str):
    vader_score = vader_model.get_sentiment([review])[0]
    bert_score = bert_model.get_sentiment([review])[0]

    return {
        "VADER": {
            "score": vader_score['compound'],
            "sentiment": "positive" if vader_score['compound'] >= 0 else "negative"
        },
        "BERT": {
            "score": bert_score,
            "sentiment": "positive" if bert_score >= 0.5 else "negative"
        }
    }

@app.post("/predict/csv")
async def predict_csv(file: UploadFile):
    df = pd.read_csv(file.file)
    df = evaluate_csv(df, vader_model, bert_model)
    return df.to_dict(orient="records")
