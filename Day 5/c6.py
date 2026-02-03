import os
import sys
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
except ImportError:
    print("spaCy library not found. Install it using: pip install spacy")
    sys.exit(1)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# Prompt for input file
print("Enter text file name: ")
filename = input()
filepath = os.path.join(sys.path[0], filename)

try:
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # 1. Display original text sample (first 300 characters)
    print("Original Text Sample:")
    print(content[:300])
    print()
    print()

    # Define the sentences to process
    sentences = [
        "All is well that ends well.",
        "Apple is looking at buying a U.K. startup for $1 billion dollars.",
        "Time flies like an arrow.",
        "The monkey ate the banana before I could stop him.",
        content  # The entire file content as the 5th sentence
    ]

    # Process each sentence
    for i, sentence in enumerate(sentences, 1):
        print(f"=== POS Tagging for Sentence {i} ===")
        print("Word\t\tPOS\t\tTag")
        print("-" * 34)

        # Process the sentence with spaCy
        doc = nlp(sentence)

        # Display POS tags for each token (excluding SPACE tokens for sentence 5)
        for token in doc:
            # Skip SPACE tokens only for sentence 5 (the file content)
            if i == 5 and token.pos_ == "SPACE":
                continue
            print(f"{token.text}\t\t{token.pos_}\t\t{token.tag_}")

        print()
        print()

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)