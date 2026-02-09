import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer


def main():
    # ============================
    # Step 0: Input text filename
    # ============================
    filename = input("Enter sports news text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load text file
    # ============================
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n=== Original Text Sample ===")
    print(documents[0][:300])
    print()

    # ============================
    # Step 2: Load spaCy model
    # ============================
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Install using: python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Preprocess text
    # Lowercasing + Stop word removal
    # ============================
    cleaned_docs = []

    for text in documents:
        doc = nlp(text.lower())
        tokens = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        cleaned_docs.append(" ".join(tokens))

    print("=== Preprocessed Text Sample ===")
    print(cleaned_docs[0][:300])
    print()

    # ============================
    # Step 4: Generate Bag-of-Words
    # ============================
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(cleaned_docs)

    # ============================
    # Step 5: Display BoW Matrix
    # ============================
    print("=== Bag-of-Words Matrix ===")
    print(bow_matrix.toarray())
    print()

    # ============================
    # Step 6: Display Word Frequencies
    # ============================
    print("=== Word Frequencies ===")
    word_freq = zip(
        vectorizer.get_feature_names(),   # ✔ FIXED LINE
        bow_matrix.toarray().sum(axis=0)
    )

    for word, freq in sorted(word_freq, key=lambda x: x[1], reverse=True):
        print(f"{word:<20} : {freq}")


if __name__ == "__main__":
    main()