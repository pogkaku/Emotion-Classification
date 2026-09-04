"""
Emotion Classification Prediction Pipeline

Expected training CSV columns:
    text, emotion

Expected test CSV column:
    text

Example:
    python main.py --train train.csv --test test.csv --output predictions.csv
"""

import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def download_nltk_resources():
    """Download the NLTK resources required by the text cleaner."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """Clean and normalize a text string."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


def train_and_predict(train_csv, test_csv, output_csv="predictions.csv"):
    """Train the Linear SVM on labelled data and predict test examples."""
    train_path = Path(train_csv)
    test_path = Path(test_csv)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_train_columns = {"text", "emotion"}
    required_test_columns = {"text"}

    if not required_train_columns.issubset(train_df.columns):
        raise ValueError(
            "Training CSV must contain the columns: text and emotion"
        )

    if not required_test_columns.issubset(test_df.columns):
        raise ValueError("Test CSV must contain the column: text")

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    model = LinearSVC(C=1)
    model.fit(X_train, train_df["emotion"])

    predictions = model.predict(X_test)

    result = test_df.copy()
    result["predicted_emotion"] = predictions
    result.to_csv(output_csv, index=False)

    print(f"Predictions saved to: {output_csv}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify text into emotion categories using Linear SVM."
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to labelled training CSV containing text and emotion columns."
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to test CSV containing a text column."
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV path. Default: predictions.csv"
    )

    args = parser.parse_args()

    train_and_predict(
        train_csv=args.train,
        test_csv=args.test,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

