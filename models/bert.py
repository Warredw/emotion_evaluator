from transformers import pipeline

class BertModel:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", truncation=True)

    def predict(self, texts):
        preds = self.classifier(texts)
        return [p['score'] if p['label'] == 'POSITIVE' else 1 - p['score'] for p in preds]
