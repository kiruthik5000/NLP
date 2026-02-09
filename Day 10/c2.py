import os
import sys
import warnings
import pandas as pd

warnings.simplefilter("ignore")


filename = "Sample.csv"
file_path = os.path.join(sys.path[0], filename)

if not os.path.exists(file_path):
    print("Dataset file not found.")
    sys.exit(0)

df = pd.read_csv(file_path)

df = df.dropna()


train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)


train_df["ft_format_binary"] = (
    "__label__" + train_df["binary_sentiment"].astype(str) + " " +
    train_df["clean_text"].astype(str)
)

print("===== FASTTEXT BINARY TRAIN DATA =====")
print(train_df["ft_format_binary"])

train_df["ft_format_binary"].to_csv(
    os.path.join(sys.path[0], "train_fasttext_bn.txt"),
    index=False,
    header=False
)


train_df["ft_format_multiclass"] = (
    "__label__" + train_df["sentiment"].astype(str) + " " +
    train_df["clean_text"].astype(str)
)

print("\n===== FASTTEXT MULTI-CLASS TRAIN DATA =====")
print(train_df["ft_format_multiclass"])

train_df["ft_format_multiclass"].to_csv(
    os.path.join(sys.path[0], "train_fasttext_mc.txt"),
    index=False,
    header=False
)



def convert_multilabels(label_string):
    labels = [lbl.strip() for lbl in str(label_string).split(",")]
    return " ".join(["__label__" + lbl for lbl in labels])

train_df["ft_format_multilabel"] = (
    train_df["emotion_labels"].apply(convert_multilabels) + " " +
    train_df["clean_text"].astype(str)
)

print("\n===== FASTTEXT MULTI-LABEL TRAIN DATA =====")
print(train_df["ft_format_multilabel"])

train_df["ft_format_multilabel"].to_csv(
    os.path.join(sys.path[0], "train_fasttext_ml.txt"),
    index=False,
    header=False
)

print("\nfastText training files generated successfully")
print("\n============================================================")
print("NOTE: fasttext module is not available")
print("Install it with: pip install fasttext")