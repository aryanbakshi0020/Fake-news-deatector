from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "data" / "raw" / "news.csv"

try:
    ENGLISH_STOPWORDS = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    ENGLISH_STOPWORDS = stopwords.words("english")


def clean_text(text):
    if text is None:
        return ""
    value = html.unescape(str(text)).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s\u0900-\u097f]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_label(value):
    if value is None:
        raise ValueError("Label cannot be None.")
    label = str(value).strip().lower()
    if label in {"real", "true", "reliable", "1", "1.0"} or label.startswith("real"):
        return "REAL"
    if label in {"fake", "false", "unreliable", "0", "0.0"} or label.startswith("fake"):
        return "FAKE"
    raise ValueError(f"Unsupported label {value!r}. Use REAL/FAKE or 1/0.")


def load_and_prepare_data(csv_path=DATASET_PATH):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].map(normalize_label)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both REAL and FAKE labels.")

    return df


def train_model(df):
    if len(df) < 8:
        raise ValueError("Please use at least 8 labeled rows for reliable model evaluation.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    min_df = 1 if len(df) < 30 else 2
    vectorizer = TfidfVectorizer(
        stop_words=ENGLISH_STOPWORDS,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=min_df,
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train_vec, y_train)

    predictions = model.predict(x_test_vec)
    labels = ["FAKE", "REAL"]
    precision = precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0),
        "report": classification_report(y_test, predictions, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
        "samples": len(df),
    }

    return model, vectorizer, metrics


def predict_news(model, vectorizer, news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    class_index = list(model.classes_).index(prediction)
    return {
        "label": "REAL NEWS" if prediction == "REAL" else "FAKE NEWS",
        "confidence": float(probabilities[class_index]),
        "probabilities": dict(zip(model.classes_, probabilities)),
    }


def explain_prediction(model, vectorizer, news_text, limit=8):
    vector = vectorizer.transform([clean_text(news_text)])
    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0] * model.coef_[0]
    ranked = sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True)
    return {
        "real": ranked[:limit],
        "fake": list(reversed(ranked[-limit:])),
    }
