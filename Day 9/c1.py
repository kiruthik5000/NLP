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