import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy

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
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Create spaCy Doc
    # ============================
    doc = nlp(content)

    # ======================================================
    # Step 3.1: Find sentences where PERSON is followed by
    #           the word "medal" in the same sentence
    # ======================================================
    print("=== Sentences with PERSON followed by 'medal' ===")
    found = False

    for sent in doc.sents:
        person_present = any(ent.label_ == "PERSON" for ent in sent.ents)
        if person_present and "medal" in sent.text.lower():
            print(f"- {sent.text.strip()}")
            found = True

    if not found:
        print("No matching sentences found.")
    print()

    # ======================================================
    # Step 3.2: Extract competition / event names
    # ======================================================
    events = []

    for ent in doc.ents:
        if ent.label_ == "EVENT":
            events.append(ent.text)
        elif ("Championship" in ent.text or
              "Run" in ent.text or
              "Grand Prix" in ent.text):
            events.append(ent.text)

    # Remove duplicates
    unique_events = set(events)

    print("=== Competitions Mentioned ===")
    if unique_events:
        for event in unique_events:
            print(f"- {event}")
    else:
        print("No events found.")

if __name__ == "__main__":
    main()