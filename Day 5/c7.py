import os
import sys
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
    import pandas as pd
except ImportError as e:
    print(f"Required library not found: {e}")
    print("Install using: pip install spacy pandas")
    sys.exit(1)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found.")
    print("Install using: python -m spacy download en_core_web_sm")
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
    print("=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    # Process specific sentence for dependency parsing
    sentence = "The dollar has hit its highest level against the euro after the Federal Reserve head said the US trade deficit is set to stabilize."
    doc_sent = nlp(sentence)

    # 2. NLP Processed Tokens with dependency parsing
    print("=== 7.2.1 NLP Processed Tokens ===")
    for token in doc_sent:
        print(f"{token.text} {token.pos_} {token.dep_}")
    print()

    # 3. Named Entities (Tuples) from the sentence
    print("=== 7.2.2 Named Entities (Tuples) ===")
    entities_tuples = [(ent.text, ent.label_) for ent in doc_sent.ents]
    print(entities_tuples)
    print()

    # 4. Named Entities DataFrame from the sentence
    print("=== 7.2.3 Named Entities DataFrame ===")
    df_sent = pd.DataFrame(entities_tuples, columns=['Entity', 'Label'])
    print(df_sent.to_string(index=True))
    print()

    # 5. First 5 Named Entities from the entire file
    print("=== 7.2.4 First 5 Named Entities from File ===")
    doc_file = nlp(content)
    file_entities = [(ent.text, ent.label_) for ent in doc_file.ents][:5]
    df_file = pd.DataFrame(file_entities, columns=['Entity', 'Label'])
    print(df_file.to_string(index=True))
    print()

    # 6. Question-based NER DataFrame
    question_text = "Taylor Swift will perform at Tokyo next Friday. Universal Music Group is releasing her new album."
    doc_question = nlp(question_text)

    print("=== Question-based NER DataFrame ===")
    question_entities = [(ent.text, ent.label_) for ent in doc_question.ents]
    df_question = pd.DataFrame(question_entities, columns=['Entity', 'Label'])
    print(df_question.to_string(index=True))
    print()

    # 7. Extracted Answers
    print("=== Extracted Answers ===")

    # Extract entities by type
    person = None
    location = None
    date = None
    organization = None

    for ent in doc_question.ents:
        if ent.label_ == "PERSON" and person is None:
            person = ent.text
        elif ent.label_ == "GPE" and location is None:
            location = ent.text
        elif ent.label_ == "DATE" and date is None:
            date = ent.text
        elif ent.label_ == "ORG" and organization is None:
            organization = ent.text

    print(f"Who is the person performing?: {person}")
    print(f"When is the performance happening?: {date}")
    print(f"Where will the concert take place?: {location}")
    print(f"Which company is releasing the album?: {organization}")
    print()

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)