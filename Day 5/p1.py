import os
import sys
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import spacy
except ImportError:
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    sys.exit(1)

filename = input().strip()
path = os.path.join(sys.path[0], filename)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines()

print("First 10 lines from the file:")
for line in lines[:10]:
    print(line)

print()

doc = nlp(content)
tokens = [token.text for token in doc[:20]]

print("First 20 tokens:")
print(tokens)
