import os
import sys
import warnings

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from nlp_utils import clean_text, split_labels

warnings.simplefilter(action='ignore')


def main():
    filename = input().strip()

    df = pd.read_csv(os.path.join(sys.path[0],filename))

    # -------------------------------
    # Validate required column
    # -------------------------------
    if 'text' not in df.columns:
        print("Column 'text' not found")
        return

    # -------------------------------
    # Data exploration
    # -------------------------------
    print("=== First 5 Rows ===")
    print(df.head())

    print()
    print(f"Number of samples: {len(df)}")

    print()
    print("=== Data Types ===")
    print(df.dtypes)

    print()
    print("=== Missing Values ===")
    print(df.isnull().sum())

    # -------------------------------
    # Text cleaning
    # -------------------------------
    df['clean_text'] = df['text'].apply(clean_text)

    print()
    print("=== Sample Cleaned Text ===")
    print(df[['text', 'clean_text']].head())

    # -------------------------------
    # TF-IDF Feature Extraction
    # -------------------------------
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    print()
    print(f"TF-IDF Shape: {tfidf_matrix.shape}")

    # -------------------------------
    # Sentiment Encoding
    # -------------------------------
    if 'sentiment' in df.columns:
        le = LabelEncoder()
        df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

        sentiment_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print()
        print("Sentiment Classes:", sentiment_mapping)

    # -------------------------------
    # Multi-label Emotion Encoding
    # -------------------------------
    if 'emotion_labels' in df.columns:
        df['emotion_list'] = df['emotion_labels'].apply(split_labels)

        mlb = MultiLabelBinarizer()
        emotion_matrix = mlb.fit_transform(df['emotion_list'])

        print()
        print("Emotion Classes:", mlb.classes_)
        print(f"Emotion Encoding Shape: {emotion_matrix.shape}")


if __name__ == "__main__":
    main()

# ----------------------------------------
import re
import string

def clean_text(text):
    """
    Cleans and normalizes input text for NLP processing.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_labels(label_text):
    """
    Converts comma-separated emotion labels into a list.
    """
    if not isinstance(label_text, str):
        return []

    labels = [label.strip() for label in label_text.split(',')]
    return [label for label in labels if label]