from nltk.sentiment.vader import SentimentIntensityAnalyzer

class VaderModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, texts):
        return [self.analyzer.polarity_scores(t) for t in texts]
