import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from models.vader import VaderModel
from models.bert import BertModel
from sklearn.metrics import classification_report



def evaluate_models(csv_path): 
    
    # load the dataset
    df = pd.read_csv(csv_path, encoding = 'latin1', sep = ";")
    
    #remove the <br /> tags
    df["review"] = df["review"].str.replace("<br />", " ", regex=False)
    
    # initialize objects of the models
    vader = VaderModel()
    bert = BertModel()
    
    # create a true label column to evaluate against
    df["true_label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    
    # make a column vader_score and bert_score
    df["vader_score"] = [s["compound"] for s in vader.get_sentiment(df["review"])]
    df["bert_score"] = bert.get_sentiment(df["review"].tolist())
    
    # make a column that converts the score into a binary prediction
    df["vader_pred"] = (df["vader_score"] >= 0).astype(int)
    df["bert_pred"] = (df["bert_score"] >= 0.5).astype(int)
    
    results = {
        "VADER Accuracy": accuracy_score(df["true_label"], df["vader_pred"]),
        "VADER F1": f1_score(df["true_label"], df["vader_pred"]),
        "BERT Accuracy": accuracy_score(df["true_label"], df["bert_pred"]),
        "BERT F1": f1_score(df["true_label"], df["bert_pred"])
    }
    
    # print class imbalance
    class_counts = df["true_label"].value_counts(normalize=True)
    print("Class Imbalance:")
    for label, count in class_counts.items():
        print(f"Label {label}: {count:.2%}")
    
    # print the results
    print("Benchmark Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    print("\nConfusion Matrices:")
    print("\nVADER Confusion Matrix:")
    print(confusion_matrix(df["true_label"], df["vader_pred"]))
    print("\nBERT Confusion Matrix:")
    print(confusion_matrix(df["true_label"], df["bert_pred"]))
    
    
    
    
    
