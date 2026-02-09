import pandas as pd
import os
import sys
import warnings
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

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

    if "binary_sentiment" in train.columns:
        binary_model = LogisticRegression(max_iter=1000)
        binary_model.fit(X_train_tfidf, train["binary_sentiment"])
        binary_preds = binary_model.predict(X_test_tfidf)

        print("\n=== Binary Classification Predictions ===")
        print(binary_preds[:10])

    if "sentiment_encoded" in train.columns:
        multi_model = MultinomialNB()
        multi_model.fit(X_train_tfidf, train["sentiment_encoded"])
        multi_preds = multi_model.predict(X_test_tfidf)

        print("\n=== Multi-Class Classification Predictions ===")
        print(multi_preds[:10])

    if "emotion_labels" in train.columns:
        train["emotion_list"] = train["emotion_labels"].apply(split_labels)
        test["emotion_list"] = test["emotion_labels"].apply(split_labels)

        mlb = MultiLabelBinarizer()
        Y_train_mlabel = mlb.fit_transform(train["emotion_list"])
        Y_test_mlabel = mlb.transform(test["emotion_list"])

        multilabel_model = OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )
        multilabel_model.fit(X_train_tfidf, Y_train_mlabel)
        multilabel_preds = multilabel_model.predict(X_test_tfidf)

        print("\n=== Multi-Label Classification Predictions ===")
        print(multilabel_preds[:5])
        print("Associated classes:", mlb.classes_)


if __name__ == "__main__":
    main()