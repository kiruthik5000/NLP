import os
import sys
import warnings
import pandas as pd

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.pipeline import EntityRuler
except ImportError:
    pass


def main():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        sys.exit(1)

    text = "The dollar has hit its highest level against the euro after the Federal Reserve head said the US trade deficit is set to stabilize."
    doc = nlp(text)

    print("=== 7.3.1 NLP Processed Tokens ===")
    for token in doc:
        print(f"{token.text} {token.pos_} {token.dep_}")
    print()

    print("=== 7.3.2 Default NER Output ===")
    default_ents = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents]
    print(pd.DataFrame(default_ents))
    print()

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "CURRENCY", "pattern": [{"LOWER": "the"}, {"LOWER": "dollar"}]},
        {"label": "CURRENCY", "pattern": [{"LOWER": "the"}, {"LOWER": "euro"}]}
    ]
    ruler.add_patterns(patterns)
    doc_single = nlp(text)

    print("=== 7.3.3 Single-Label Rule-based Matching Output ===")
    single_ents = [{"Entity": ent.text, "Label": ent.label_} for ent in doc_single.ents]
    print(pd.DataFrame(single_ents))
    print()

    ruler.add_patterns([{"label": "ECONOMIC_TERM", "pattern": [{"LOWER": "trade"}, {"LOWER": "deficit"}]}])
    doc_multi = nlp(text)

    print("=== 7.3.4 Multi-Label Rule-based Matching Output ===")
    multi_ents = [{"Entity": ent.text, "Label": ent.label_} for ent in doc_multi.ents]
    print(pd.DataFrame(multi_ents))
    print()

    print("=== 7.3.5 Comparison Summary ===")
    print("\nDefault NER Entities:")
    print(pd.DataFrame(default_ents))
    print("\nNER + Single-Label Rule-based Entities:")
    print(pd.DataFrame(single_ents))
    print("\nNER + Multi-Label Rule-based Entities:")
    print(pd.DataFrame(multi_ents))

    print("\nObservation:")
    print("- Default NER may miss domain-specific entities like currencies.")
    print("- Rule-based matching accurately captures predefined terms.")
    print("- Multi-label matching provides richer, task-specific entity extraction.")


if __name__ == "__main__":
    main()