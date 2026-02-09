import pandas as pd
import os
import sys
import warnings

warnings.simplefilter("ignore")

from nlp_utils import clean_text, split_labels, split_data

def main():
    train_file = "Sample.csv"
    test_file  = "Sample.csv"

    train_path = os.path.join(sys.path[0], train_file)
    test_path  = os.path.join(sys.path[0], test_file)

    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    train_df = train_df.dropna()
    test_df  = test_df.dropna()

    print("\n=== Training Data Sample ===")
    print(train_df.head())
    print("\n=== Test Data Sample ===")
    print(test_df.head())

    if "clean_text" not in train_df.columns:
        if "review" in train_df.columns:
            train_df["clean_text"] = train_df["review"].apply(clean_text)
            test_df["clean_text"]  = test_df["review"].apply(clean_text)
        else:
            print("No column 'clean_text' or 'review' found.")
            sys.exit(1)

    if "binary_sentiment" in train_df.columns:
        train_df['ft_label_binary'] = '__label__' + train_df['binary_sentiment'].astype(str)
        train_df['ft_format_binary'] = train_df['ft_label_binary'] + " " + train_df['clean_text']
        print("\n=== Binary FastText Sample ===")
        print("\n".join(train_df['ft_format_binary'].head().tolist()))

        out_path = os.path.join(sys.path[0], "train_fasttext_bn.txt")
        train_df['ft_format_binary'].to_csv(out_path, index=False, header=False)

    if "sentiment" in train_df.columns:
        train_df['ft_label_multiclass'] = '__label__' + train_df['sentiment'].astype(str)
        test_df['ft_label_multiclass']  = '__label__' + test_df['sentiment'].astype(str)

        train_df['ft_format_multiclass'] = train_df['ft_label_multiclass'] + " " + train_df['clean_text']
        test_df['ft_format_multiclass']  = test_df['ft_label_multiclass'] + " " + test_df['clean_text']

        print("\n=== Multi-Class FastText Sample ===")
        print("\n".join(train_df['ft_format_multiclass'].head().tolist()))

        out_path_mc = os.path.join(sys.path[0], "train_fasttext_mc.txt")
        train_df['ft_format_multiclass'].to_csv(out_path_mc, index=False, header=False)

    if "emotion_labels" in train_df.columns:
        def convert_labels(row):
            labels = split_labels(row['emotion_labels'])
            labels = ['__label__' + l for l in labels]
            return " ".join(labels)

        train_df['ft_label_multi'] = train_df.apply(convert_labels, axis=1)
        train_df['ft_format_multi'] = train_df['ft_label_multi'] + " " + train_df['clean_text']

        print("\n=== Multi-Label FastText Sample ===")
        print("\n".join(train_df['ft_format_multi'].head().tolist()))

        out_path_ml = os.path.join(sys.path[0], "train_fasttext_ml.txt")
        train_df['ft_format_multi'].to_csv(out_path_ml, index=False, header=False)


if __name__ == "__main__":
    main()

# ----------------------------------------

import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Basic text cleaning:
    - Lowercase
    - Remove punctuation/special characters
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_labels(label_str):
    """
    Split comma-separated labels into a list.
    Handles NaN values safely.
    """
    if pd.isna(label_str):
        return []
    return [l.strip() for l in label_str.split(",")]


def split_data(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets.
    Returns train_df, test_df with reset indexes.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)