import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import spacy
import sys
text = open(os.path.join(sys.path[0], "Sample.txt")).read()
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print(doc.text)
print(doc[:20])
print(f"Total number of tokens: {len(doc)}")