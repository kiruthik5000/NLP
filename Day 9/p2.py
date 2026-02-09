import pandas as pd
import os
import sys
import warnings
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

warnings.simplefilter("ignore")

from nlp_utils import clean_text, split_labels, split_data


def main():
    f143 = input("Enter dataset filename (CSV or Excel): ").strip()
    file_path = os.path.join(sys.path[0], f143)

    try:
        if f143.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print("\n=== First 5 Rows ===")
    print(df.head())
    print(f"\nNumber of samples: {df.shape[0]}")
    print("\n=== Data Types ===")
    print(df.dtypes)

    train, test = split_data(df)
    print(f"\nTrain: {train.shape[0]}, Test: {test.shape[0]}")

    if "review" not in train.columns:
        print("Column 'review' not found — cannot clean text.")
        sys.exit(1)

    train["cleaned_text"] = train["review"].apply(clean_text)
    test["cleaned_text"] = test["review"].apply(clean_text)

    print("\n=== Sample Cleaned Text ===")
    print(train[["review", "cleaned_text"]].head())

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(train["cleaned_text"])
    X_test_tfidf = tfidf.transform(test["cleaned_text"])
    print("\n=== TF-IDF Shapes ===")
    print("Train:", X_train_tfidf.shape, "Test:", X_test_tfidf.shape)

    tfidf_file = os.path.join(sys.path[0], "tfidf.pkl")
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidf, f)

    if "sentiment" in train.columns:
        le_sentiment = LabelEncoder()
        train["sentiment_encoded"] = le_sentiment.fit_transform(train["sentiment"])
        test["sentiment_encoded"] = le_sentiment.transform(test["sentiment"])
        print("\n=== Sentiment Mapping ===")
        print(dict(zip(le_sentiment.classes_, le_sentiment.transform(le_sentiment.classes_))))

    if "emotion_labels" in train.columns:
        train["emotion_list"] = train["emotion_labels"].apply(split_labels)
        test["emotion_list"] = test["emotion_labels"].apply(split_labels)

        mlb = MultiLabelBinarizer()
        Y_train_mlabel = mlb.fit_transform(train["emotion_list"])
        Y_test_mlabel = mlb.transform(test["emotion_list"])
        print("\n=== Multi-label Classes ===")
        print(mlb.classes_)
        print("Multi-label shape (train):", Y_train_mlabel.shape)


if __name__ == "__main__":
    main()

# ------------------------------------------

import pandas as pd
import re

ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by do does doing down during each few for from further had
has have having he her here hers him himself his how i if in into is it its itself me
more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why with you your yours yourself yourselves
""".split())

def clean_text(text):
    """
    Lowercase, remove URLs, mentions, punctuation, numbers, stopwords
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in ENGLISH_STOPWORDS]
    return " ".join(words)

def split_labels(label_string):
    if pd.isna(label_string) or label_string == "":
        return []
    return [l.strip() for l in label_string.split(",")]

def split_data(df, test_ratio=0.2, random_state=42):
    train = df.sample(frac=1 - test_ratio, random_state=random_state)
    test = df.drop(train.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)