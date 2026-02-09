import os, sys
import warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spacy

nlp = spacy.load('en_core_web_sm')

filename = os.path.join(sys.path[0], input("Enter text file name:").strip())

content = open(filename).read()

doc = nlp(content)

res = {"PERSON": 0, "GPE": 0, "DATE": 0}

print("\n=== Original Text Sample (First 300 chars) ===")
print(content[:300])
print()

print("=== Named Entities (PERSON, GPE, DATE) ===")
for ent in doc.ents:
    if ent.label_ in ['PERSON', 'GPE', 'DATE']:
        print(f"{ent.text} ({ent.label_})")
        res[ent.label_] += 1

print()

print("=== Entity Frequency ===")
for ent in ['PERSON', 'DATE','GPE']:
    print(f"{ent}: {res[ent]}")