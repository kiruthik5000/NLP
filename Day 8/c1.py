import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from sklearn.feature_extraction.text import CountVectorizer


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
    # Task 8.1.2 – Sentence BoW
    # =====================================================
    sentence = "LOL LOL slay slay slay queen."

    doc = nlp(sentence.lower())
    tokens = [
        token.text
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    clean_sentence = [" ".join(tokens)]

    print("Cleaned Sentence:")
    print(clean_sentence[0])

    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(clean_sentence)

    print("\nBoW Word Frequencies (Sentence):")
    for word, index in vectorizer.vocabulary_.items():
        print(word, ":", bow[0, index])

    # =====================================================
    # Task 8.1.3 – Business News File BoW
    # =====================================================
    filename = "Sample.txt"
    file_path = os.path.join(sys.path[0], filename)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
    except:
        print("File not found")
        sys.exit(1)

    cleaned_docs = []
    for text in documents:
        doc = nlp(text.lower())
        tokens = [
            token.text
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        cleaned_docs.append(" ".join(tokens))

    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(cleaned_docs)

    print("\nBoW Word Frequencies (Business Data):")
    word_counts = bow.sum(axis=0)

    for word, index in vectorizer.vocabulary_.items():
        print(word, ":", int(word_counts[0, index]))


if __name__ == "__main__":
    main()