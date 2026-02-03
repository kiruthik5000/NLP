import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentence = "Claxton hunting first major medal"
doc = nlp(sentence)

subject = ""
main_verb = ""

for token in doc:
    # Find the main verb (ROOT of the sentence)
    if token.dep_ == "ROOT":
        main_verb = token.text
    # Find the nominal subject
    if "subj" in token.dep_: # This covers nsubj, csubj, agent, etc.
        subject = token.text
    # Specific for nsubj for direct subject
    if token.dep_ == "nsubj":
        subject = token.text


print(f"Sentence : {sentence}")
print(f"Subject  : {subject}")
print(f"Main Verb: {main_verb}")

# Example from the user's prompt (second sentence)
sentence_2 = "British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid."
doc_2 = nlp(sentence_2)

subject_2 = ""
main_verb_2 = ""

for token in doc_2:
    if token.dep_ == "ROOT":
        main_verb_2 = token.text
    if token.dep_ == "nsubj":
        subject_2 = token.text

print(f"\nSentence : {sentence_2}")
print(f"Subject  : {subject_2}")
print(f"Main Verb: {main_verb_2}")
