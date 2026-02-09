import os
import sys
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError:
    print("spaCy library not found. Install it using: pip install spacy")
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

print("Enter text file name: ", end='')
filename = input()
filepath = os.path.join(sys.path[0], filename)

try:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        doc = nlp(content)

    stop_words = STOP_WORDS.copy()

    custom_add = {"officially", "announced", "present", "run"}
    for word in custom_add:
        stop_words.add(word)

    custom_remove = {"hence", "every", "he"}
    for word in custom_remove:
        stop_words.discard(word)

    filtered_tokens = []
    for token in doc:
        if (token.text.lower() not in stop_words and
                not token.is_punct and
                not token.is_space):
            filtered_tokens.append(token.lemma_.lower())

    print(f"Filtered Tokens (First 20):")
    print(filtered_tokens[:20])
    print()

    cleaned_text = ' '.join(filtered_tokens)

    print("Cleaned Text Sample:")
    print(cleaned_text[:200])
    print()

except FileNotFoundError:
    print("Error: File not found")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)