import os
import sys
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def main():
    f143 = input("").strip()
    file_path = os.path.join(sys.path[0], f143)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{f143}' not found.")
        sys.exit(1)

    print("First 5 rows from the file:")
    print(df.head())
    print()
    train_df = df.copy()
    test_df = df.copy()

    train_df['clean_text'] = train_df['clean_text'].fillna('')
    test_df['clean_text'] = test_df['clean_text'].fillna('')

    train_df = train_df[train_df['clean_text'].str.strip() != '']
    test_df = test_df[test_df['clean_text'].str.strip() != '']

    train_df['clean_text'] = train_df['clean_text'].str.lower()
    test_df['clean_text'] = test_df['clean_text'].str.lower()


    label_mapping = {label: idx for idx, label in enumerate(df['sentiment'].unique())}
    train_df['sentiment_encoded'] = train_df['sentiment'].map(label_mapping)
    test_df['sentiment_encoded'] = test_df['sentiment'].map(label_mapping)
    y_true = test_df['sentiment_encoded']

    random.seed(42)
    def fasttext_predict_mock(text):
        return random.choice(df['sentiment'].unique())
    test_df['pred_ft'] = test_df['clean_text'].apply(fasttext_predict_mock)

    multi_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", MultinomialNB())
    ])
    multi_pipeline.fit(train_df["clean_text"], train_df["sentiment_encoded"])
    test_df['pred_sklearn'] = multi_pipeline.predict(test_df["clean_text"])

    labels = df['sentiment'].unique().tolist()
    random.seed(42)
    def genai_predict_mock(text):
        return random.choice(labels)
    test_df['pred_genai'] = test_df['clean_text'].apply(genai_predict_mock)

    ft_acc   = accuracy_score(y_true, test_df['pred_ft'].map(label_mapping))
    sk_acc   = accuracy_score(y_true, test_df['pred_sklearn'])
    gen_acc  = accuracy_score(y_true, test_df['pred_genai'].map(label_mapping))

    print("fastText Accuracy: ", round(ft_acc, 4))
    print("sklearn Accuracy: ", round(sk_acc, 4))
    print("GenAI Accuracy:   ", round(gen_acc, 4))
    print()

    test_df['agree_ft_sk']   = (test_df['pred_ft'] == test_df['pred_sklearn'].map({v:k for k,v in label_mapping.items()}))
    test_df['agree_ft_gen']  = (test_df['pred_ft'] == test_df['pred_genai'])
    test_df['agree_sk_gen']  = (test_df['pred_sklearn'].map({v:k for k,v in label_mapping.items()}) == test_df['pred_genai'])

    alignment_results = test_df[['agree_ft_sk', 'agree_ft_gen', 'agree_sk_gen']].mean()
    print("Alignment Results:")
    print(alignment_results)
    print()


    test_df['ft_pred_raw'] = test_df['clean_text'].apply(lambda x: (fasttext_predict_mock(x), random.random()))
    print("Sample fastText raw predictions:")
    print(test_df[['clean_text', 'ft_pred_raw']].head(5))
    print()


    print(" Where fastText > sklearn?")
    print(test_df[(test_df['agree_ft_gen']) & (~test_df['agree_ft_sk'])].head(3))
    print()

    print(" Where sklearn > fastText?")
    print(test_df[(~test_df['agree_ft_sk']) & (test_df['agree_sk_gen'])].head(3))
    print()

    print(" Where GenAI > both?")
    print(test_df[
        (test_df['pred_genai'].map(label_mapping) == y_true) &
        (test_df['pred_sklearn'] != y_true) &
        (test_df['pred_ft'].map(label_mapping) != y_true)
    ].head(3))


if __name__ == "__main__":
    main()