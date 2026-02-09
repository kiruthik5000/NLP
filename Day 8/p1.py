import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    # ============================
    # Step 0: Input text filename
    # ============================
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load text file
    # ============================
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    # ============================
    # Step 2: Load spaCy model
    # ============================
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Install spaCy model using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Text Preprocessing
    # (Lowercase + Stopword Removal)
    # ============================
    doc = nlp(content.lower())

    cleaned_tokens = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_space and token.is_alpha
    ]

    cleaned_text = " ".join(cleaned_tokens)

    print("=== Cleaned Text Sample ===")
    print(" ".join(cleaned_tokens[:50]))
    print()

    # ============================
    # Step 4: TF-IDF Vectorization
    # ============================
    documents = [cleaned_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # FIX FOR OLD SCIKIT-LEARN
    feature_names = vectorizer.get_feature_names()
    idf_values = vectorizer.idf_

    # ============================
    # Step 5: Display Results
    # ============================
    print("=== TF-IDF Features ===")
    print(feature_names)
    print()

    print("=== IDF Values ===")
    for word, idf in zip(feature_names, idf_values):
        print(f"{word:20s} : {idf:.4f}")
    print()

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    print("=== TF-IDF Matrix ===")
    print(tfidf_df.round(4))

if __name__ == "__main__":
    main()