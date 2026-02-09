import pandas as pd
import os
import sys
import warnings

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


warnings.simplefilter("ignore")

from nlp_utils import clean_text, split_labels, split_data


def main():

    filename = input("Enter dataset filename (CSV or Excel): ").strip()
    file_path = os.path.join(sys.path[0], filename)

    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        print("Only CSV or Excel files supported")
        sys.exit(1)

    print("\n=== Dataset Preview ===")
    print(df.head())

    train, test = split_data(df)

    train["clean_text"] = train["text"].apply(clean_text)
    test["clean_text"] = test["text"].apply(clean_text)

    print("\n===== Binary Classification =====")

    binary_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    binary_pipeline.fit(train["clean_text"], train["binary_sentiment"])
    print("Binary Predictions:", binary_pipeline.predict(test["clean_text"])[:10])

    print("\n===== Multi-Class Classification =====")

    le = LabelEncoder()
    train["sentiment_encoded"] = le.fit_transform(train["sentiment"])
    test["sentiment_encoded"] = le.transform(test["sentiment"])

    multi_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", MultinomialNB())
    ])

    multi_pipeline.fit(train["clean_text"], train["sentiment_encoded"])
    print("Multi-Class Predictions:", multi_pipeline.predict(test["clean_text"])[:10])

    print("\n===== Multi-Label Classification =====")

    train["emotion_list"] = train["emotion_labels"].apply(split_labels)
    test["emotion_list"] = test["emotion_labels"].apply(split_labels)

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train["emotion_list"])

    multilabel_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    multilabel_pipeline.fit(train["clean_text"], Y_train)
    print("Multi-Label Predictions:", multilabel_pipeline.predict(test["clean_text"])[:5])
    print("Classes:", mlb.classes_)


if __name__ == "__main__":
    main()

# ---------------------------------------------
# ==============================
# nlp_utils.py
# ==============================

import pandas as pd
import re

# -----------------------------
# Built-in English stopwords
# -----------------------------
ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by do does doing down during each few for from further had
has have having he her here hers him himself his how i if in into is it its itself me
more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why with you your yours yourself yourselves
""".split())


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in ENGLISH_STOPWORDS]
    return " ".join(words)


# -----------------------------
# Multi-label splitter
# -----------------------------
def split_labels(label_string):
    if pd.isna(label_string) or label_string == "":
        return []
    return [l.strip() for l in label_string.split(",")]


# -----------------------------
# Train/Test split
# -----------------------------
def split_data(df, test_ratio=0.2, random_state=42):
    train = df.sample(frac=1 - test_ratio, random_state=random_state)
    test = df.drop(train.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)