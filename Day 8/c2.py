import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    # ============================
    # Load spaCy model
    # ============================
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("Install spaCy model using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    # =====================================================
    # Demo 8.1.3 – Read & Clean Documents
    # =====================================================
    filename = "Sample.txt"
    file_path = os.path.join(sys.path[0], filename)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
    except:
        print("File not found")
        sys.exit(1)

    clean_docs = []
    for text in documents:
        doc = nlp(text.lower())
        tokens = [
            token.text
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        clean_docs.append(" ".join(tokens))

    # =====================================================
    # TF-IDF Vectorization
    # =====================================================
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_docs)

    # ✅ FIX HERE
    features = tfidf_vectorizer.get_feature_names()
    idf_values = tfidf_vectorizer.idf_

    print("\nTF-IDF Breakdown (Multiple Documents):")

    for doc_idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        print(f"\nDocument {doc_idx + 1}:")
        for word_idx, tfidf in enumerate(doc_vector):
            if tfidf > 0:
                word = features[word_idx]
                idf = idf_values[word_idx]
                tf = tfidf / idf
                print(
                    f"{word:12s} | "
                    f"TF: {tf:.4f} | "
                    f"IDF: {idf:.4f} | "
                    f"TF-IDF: {tfidf:.4f}"
                )

    # =====================================================
    # Observations & Interpretations
    # =====================================================
    print("\nKey Observations & Interpretations:")
    print("1. Common words across documents have lower IDF values.")
    print("2. Rare words receive higher TF-IDF scores.")
    print("3. TF-IDF highlights important document-specific terms.")
    print("4. It improves document comparison over simple BoW.")


if __name__ == "__main__":
    main()