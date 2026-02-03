import os, sys
import warnings

warnings.simplefilter(action='ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import spacy
except:
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    sys.exit(1)

filepath = os.path.join(sys.path[0], input().strip())

content = open(filepath).read()

lines = content.splitlines()

print("First 10 lines from the file:")
for line in lines[:10]:
    print(line)

doc = nlp(content)

tokens = [token.text for token in doc]

print("First 20 tokens:")
print(tokens[:20])
print()

print("POS Tagging Output:")
print("Word\tPOS\tTag")
print('-' * 30)

for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.tag_}")