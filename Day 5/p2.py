import os
import sys
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import spacy
from nltk.stem import SnowballStemmer

filename = input("Enter text file name: ").strip()
path = os.path.join(sys.path[0], filename)

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language='english')

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

print("Original Text Sample:")
print(content[:300])
print()

doc = nlp(content)

tokens = [token for token in doc if not token.is_space]

print(f"Total Tokens Count: {len(tokens)}")
print()

lemmas = [token.lemma_ for token in tokens]
stems = [stemmer.stem(token.text.lower()) for token in tokens]

print("=== Lemmatized Sample (First 20 tokens) ===")
print(lemmas[:20])
print()

print("Word --> Lemma")
for word, lemma in zip(tokens[:30], lemmas[:30]):
    print(f"{word.text} --> {lemma}")
print()

print("=== Stemmed Sample (First 20 tokens) ===")
print(stems[:20])
print()

print("Word --> Stem")
for word, stem in zip(tokens[:30], stems[:30]):
    print(f"{word.text} --> {stem}")
print()

print("=== Comparison: Lemmatization vs Stemming ===")
print("Word\t\tLemma\t\tStem")
print("------------------------------------------")

for w, l, s in zip(tokens[:30], lemmas[:30], stems[:30]):
    print(f"{w.text}\t\t{l}\t\t{s}")

print()
print("Conclusion:")
print(
    "Lemmatization produces dictionary-based meaningful root words, "
    "while stemming may distort words by chopping suffixes. "
    "For NLP tasks like search, topic modeling, and information retrieval, "
    "lemmatization gives better and cleaner output."
)
