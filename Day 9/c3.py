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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.simplefilter("ignore")

from nlp_utils import clean_text, split_labels, split_data

def main():

    filename = input("Enter dataset filename (CSV or Excel): ").strip()
    f143 = os.path.join(sys.path[0], filename)

    if filename.endswith(".csv"):
        df = pd.read_csv(f143)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(f143, engine="openpyxl")
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
    binary_preds = binary_pipeline.predict(test["clean_text"])
    print("Binary Predictions:", binary_preds[:10])

    binary_true = test["binary_sentiment"]
    print("\nBinary Accuracy:", accuracy_score(binary_true, binary_preds))
    print("\nBinary Classification Report:\n", classification_report(binary_true, binary_preds))
    print("Binary Confusion Matrix:\n", confusion_matrix(binary_true, binary_preds))

    print("\n===== Multi-Class Classification =====")
    le = LabelEncoder()
    train["sentiment_encoded"] = le.fit_transform(train["sentiment"])
    test["sentiment_encoded"] = le.transform(test["sentiment"])

    multi_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", MultinomialNB())
    ])

    multi_pipeline.fit(train["clean_text"], train["sentiment_encoded"])
    multi_preds = multi_pipeline.predict(test["clean_text"])
    print("Multi-Class Predictions:", multi_preds[:10])
    multi_true = test["sentiment_encoded"]
    print("\nMulti-Class Accuracy:", accuracy_score(multi_true, multi_preds))
    print("\nMulti-Class Classification Report:\n", classification_report(multi_true, multi_preds))
    print("Multi-Class Confusion Matrix:\n", confusion_matrix(multi_true, multi_preds))

    print("\n===== Multi-Label Classification =====")
    train["emotion_list"] = train["emotion_labels"].apply(split_labels)
    test["emotion_list"] = test["emotion_labels"].apply(split_labels)

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train["emotion_list"])
    Y_test_mlabel = mlb.transform(test["emotion_list"])

    multilabel_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    multilabel_pipeline.fit(train["clean_text"], Y_train)
    multi_label_preds = multilabel_pipeline.predict(test["clean_text"])
    print("Multi-Label Predictions (first 5 rows):\n", multi_label_preds[:5])
    print("Classes:", mlb.classes_)

    micro_f1 = f1_score(Y_test_mlabel, multi_label_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_test_mlabel, multi_label_preds, average="macro", zero_division=0)
    per_label_f1 = f1_score(Y_test_mlabel, multi_label_preds, average=None, zero_division=0)

    print("\nMulti-Label Micro F1 Score:", micro_f1)
    print("Multi-Label Macro F1 Score:", macro_f1)
    print("\nPer-Label F1 Scores:")
    for emotion, score in zip(mlb.classes_, per_label_f1):
        print(f"{emotion}: {score:.4f}")

    print("\n========== SUMMARY ==========")
    print("Binary Accuracy:", accuracy_score(binary_true, binary_preds))
    print("Multi-Class Accuracy:", accuracy_score(multi_true, multi_preds))
    print("Multi-Label Micro F1:", micro_f1)
    print("Multi-Label Macro F1:", macro_f1)


if __name__ == "__main__":
    main()

# -----------------------------------------------
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
