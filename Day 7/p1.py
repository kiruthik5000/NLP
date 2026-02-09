import os, sys
import warnings

warnings.simplefilter(action='ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlp.vocab)

filepath = os.path.join(sys.path[0], input("Enter text file name:").strip())

content = open(filepath).read()

doc_main = nlp(content)

atheletes = ["Sarah Claxton", "Sonia O'Sullivan", "Irina Shevchenko"]
sports = ["European Indoor Championships", "World Cross Country Championships", "London marathon",
          "Bupa Great Ireland Run"]
athlete_patterns = [nlp.make_doc(word) for word in atheletes]

matcher.add("ATH", athlete_patterns)

sports_patterns = [nlp.make_doc(sport) for sport in sports]

matcher.add("SPORT", sports_patterns)

matches = matcher(doc_main)

print("\n=== Original Text Sample (First 300 chars) ===")
print(content[:300])
print()

res = {}

for match_id, start, end in matches:
    match_name = nlp.vocab.strings[match_id]
    res.setdefault(match_name, []).append(doc_main[start:end])

i = 0
print("=== Matched Athlete Names ===")
for val in res["ATH"]:
    print(f"- {val}")
print()

print("=== Matched Sports Events ===")
for val in res["SPORT"]:
    print(f"- {val}")

print()