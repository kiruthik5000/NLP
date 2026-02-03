import os
import sys
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
    from nltk.stem import SnowballStemmer
except ImportError as e:
    print(f"Required library not found: {e}")
    print("Install using: pip install spacy nltk")
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

stemmer = SnowballStemmer(language='english')

print("Enter text file name for full text processing: ")
filename = input()
filepath = os.path.join(sys.path[0], filename)

try:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    print("Original Text Sample:")
    print(content[:300])
    print()

    print("=== Lemmatization: Individual Words ===")
    sample_words = "friendship studied was am is organizing matches"
    doc_sample = nlp(sample_words)
    for token in doc_sample:
        if not token.is_space:
            print(f"{token.text} -> {token.lemma_}")
    print()

    print("=== Stemming: Individual Words ===")
    for word in sample_words.split():
        stem = stemmer.stem(word)
        print(f"{word} --> {stem}")
    print()

    print("=== Lemmatization: Full Text ===")
    doc_full = nlp(content)
    tokens_full = [token for token in doc_full if not token.is_space]
    for token in tokens_full[:50]:
        print(f"{token.text} --> {token.lemma_}")
    print()

    print("=== Stemming: Full Text ===")
    for token in tokens_full[:50]:
        stem = stemmer.stem(token.text.lower())
        print(f"{token.text} --> {stem}")
    print()

    print("=== Practice 6.2: Lemmatization vs Stemming ===")
    print("Word\t\tLemma\t\tStem")
    print("-" * 42)

    practice_words = "running good universities flies fairer is"
    doc_practice = nlp(practice_words)
    practice_tokens = [token for token in doc_practice if not token.is_space]

    for token in practice_tokens:
        lemma = token.lemma_
        stem = stemmer.stem(token.text.lower())
        print(f"{token.text}\t\t{lemma}\t\t{stem}")
    print()

    print("Conclusion:")
    print(
        "Lemmatization produces dictionary-based meaningful root words, while stemming may distort words by chopping suffixes. For NLP tasks like search, topic modeling, and information retrieval, lemmatization gives better and cleaner output.")
    print()

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)