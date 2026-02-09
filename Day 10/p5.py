import pandas as pd
import os
import sys
import warnings
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.simplefilter("ignore")

from nlp_utils import clean_text, split_labels, split_data


def main():

    filename = sys.argv[1] if len(sys.argv) > 1 else "Sample.csv"
    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print("Dataset not found.")
        return

    df = pd.read_csv(file_path) if filename.endswith(".csv") else pd.read_excel(file_path)

    train, test = split_data(df)


    train["clean_text"] = train["review"].apply(clean_text)
    test["clean_text"] = test["review"].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X_train = tfidf.fit_transform(train["clean_text"])
    X_test = tfidf.transform(test["clean_text"])

    if "binary_sentiment" in train.columns:

        y_train = train["binary_sentiment"].map({"negative": 0, "positive": 1})
        y_test = test["binary_sentiment"].map({"negative": 0, "positive": 1})

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("\n===== BINARY MODEL =====")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds, zero_division=0))

    if "sentiment_encoded" in train.columns:

        model = MultinomialNB()
        model.fit(X_train, train["sentiment_encoded"])
        preds = model.predict(X_test)

        print("\n===== MULTI-CLASS MODEL =====")
        print("Accuracy:", accuracy_score(test["sentiment_encoded"], preds))
        print(confusion_matrix(test["sentiment_encoded"], preds))
        print(classification_report(test["sentiment_encoded"], preds, zero_division=0))

    if "emotion_labels" in train.columns:

        train["emotion_list"] = train["emotion_labels"].apply(split_labels)
        test["emotion_list"] = test["emotion_labels"].apply(split_labels)

        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(train["emotion_list"])
        Y_test = mlb.transform(test["emotion_list"])

        model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        print("\n===== MULTI-LABEL MODEL =====")
        print("Micro F1:", f1_score(Y_test, preds, average="micro", zero_division=0))
        print("Macro F1:", f1_score(Y_test, preds, average="macro", zero_division=0))
        print(classification_report(Y_test, preds, target_names=mlb.classes_, zero_division=0))


if __name__ == "__main__":
    main()
# -------------------------------------------------
import pandas as pd
import re
#s
ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by do does doing down during each few for from further had
has have having he her here hers him himself his how i if in into is it its itself me
more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where
which while who whom why with you your yours yourself yourselves
""".split())

def clean_text(text):
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