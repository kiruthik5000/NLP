import pandas as pd
import os
import sys
import warnings

warnings.simplefilter("ignore")

def main():

    filename = "Sample.csv"
    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print("Dataset file not found.")
        return

    df = pd.read_csv(file_path)

    df = df.dropna()

    train_df = df.sample(frac=0.8, random_state=42)
    test_df  = df.drop(train_df.index)

    train_df["ft_label_binary"] = "__label__" + train_df["binary_sentiment"].astype(str)
    train_df["ft_format_binary"] = (
        train_df["ft_label_binary"] + " " + train_df["clean_text"]
    )

    print("\n===== FASTTEXT BINARY TRAIN DATA =====")
    print(train_df["ft_format_binary"])

    train_df["ft_format_binary"].to_csv(
        os.path.join(sys.path[0], "train_fasttext_bn.txt"),
        index=False,
        header=False
    )

    train_df["ft_label_multiclass"] = "__label__" + train_df["sentiment"].astype(str)
    train_df["ft_format_multiclass"] = (
        train_df["ft_label_multiclass"] + " " + train_df["clean_text"]
    )

    print("\n===== FASTTEXT MULTI-CLASS TRAIN DATA =====")
    print(train_df["ft_format_multiclass"])

    train_df["ft_format_multiclass"].to_csv(
        os.path.join(sys.path[0], "train_fasttext_mc.txt"),
        index=False,
        header=False
    )

    def convert_multilabels(label_str):
        labels = label_str.split(",")
        labels = ["__label__" + l.strip() for l in labels]
        return " ".join(labels)

    train_df["ft_label_multilabel"] = train_df["emotion_labels"].apply(convert_multilabels)
    train_df["ft_format_multilabel"] = (
        train_df["ft_label_multilabel"] + " " + train_df["clean_text"]
    )

    print("\n===== FASTTEXT MULTI-LABEL TRAIN DATA =====")
    print(train_df["ft_format_multilabel"])

    train_df["ft_format_multilabel"].to_csv(
        os.path.join(sys.path[0], "train_fasttext_ml.txt"),
        index=False,
        header=False
    )

    print("\n fastText training files generated successfully")


if __name__ == "__main__":
    main()