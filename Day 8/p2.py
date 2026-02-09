# ============================================================
# CRT: Word Embeddings and Semantic Similarity Analysis
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
import spacy
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Suppress warnings
# ------------------------------------------------------------
warnings.simplefilter(action='ignore')

# ------------------------------------------------------------
# Load spaCy Model
# ------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Install using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# ------------------------------------------------------------
# File Input
# ------------------------------------------------------------
filename = input("Enter sports news text file name: ")

file_path = os.path.join(sys.path[0], filename)

if not os.path.exists(file_path):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

# ------------------------------------------------------------
# Read File
# ------------------------------------------------------------
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# ------------------------------------------------------------
# Original Text Sample
# ------------------------------------------------------------
print("\n=== Original Text Sample (First 300 chars) ===")
print(content[:300])
print()

# ------------------------------------------------------------
# Split Documents
# ------------------------------------------------------------
documents = [doc.strip() for doc in content.split("----") if doc.strip()]

# ------------------------------------------------------------
# Text Preprocessing
# ------------------------------------------------------------
clean_docs = []
clean_tokens = []

for doc in documents:
    spacy_doc = nlp(doc.lower())
    tokens = [
        token.text for token in spacy_doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]
    clean_docs.append(" ".join(tokens))
    clean_tokens.extend(tokens)

# ------------------------------------------------------------
# Cleaned Text Sample
# ------------------------------------------------------------
print("=== Cleaned Text Sample ===")
print(clean_tokens[:50])
print()

# ------------------------------------------------------------
# Bag-of-Words
# ------------------------------------------------------------
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(clean_docs)

# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_docs)

# ------------------------------------------------------------
# TF-IDF Features
# ------------------------------------------------------------
print("=== TF-IDF Features ===")
features = tfidf_vectorizer.get_feature_names()
print(features)
print()

# ------------------------------------------------------------
# IDF Values
# ------------------------------------------------------------
print("=== IDF Values ===")
idf_values = tfidf_vectorizer.idf_
for word, value in zip(features, idf_values):
    print(f"{word} : {value:.4f}")
print()

# ------------------------------------------------------------
# TF-IDF Matrix
# ------------------------------------------------------------
print("=== TF-IDF Matrix ===")
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=features
).round(4)
print(tfidf_df)
print()

# ============================================================
# Task 3.1: Word Embeddings
# ============================================================
embed_vectors = np.array([nlp(doc).vector for doc in clean_docs])

print("=== Word Embedding Vectors ===")
print(embed_vectors)
print()

# ============================================================
# Task 3.2: Vector Shapes
# ============================================================
print("=== Vector Shapes ===")
print("BoW shape:", bow_matrix.shape)
print("TF-IDF shape:", tfidf_matrix.shape)
print("Embedding shape:", embed_vectors.shape)
print()

# ============================================================
# Task 3.3: Cosine Similarity
# ============================================================
print("=== Cosine Similarity (BoW) ===")
print(cosine_similarity(bow_matrix.toarray()))
print()

print("=== Cosine Similarity (TF-IDF) ===")
print(cosine_similarity(tfidf_matrix.toarray()))
print()

print("=== Cosine Similarity (Embeddings) ===")
print(cosine_similarity(embed_vectors))
print()

# ============================================================
# Task 3.4: Observations
# ============================================================
print("=== Observations ===")
print(
    "1. Bag-of-Words considers only word frequency.\n"
    "2. TF-IDF highlights important words across documents.\n"
    "3. Word embeddings capture semantic meaning and context.\n"
    "4. Embedding similarity reflects deeper relationships "
    "between sports news articles."
)