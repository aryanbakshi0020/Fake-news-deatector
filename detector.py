import html
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "news.csv"

try:
    ENGLISH_STOPWORDS = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    ENGLISH_STOPWORDS = stopwords.words("english")


def clean_text(text):
    text = html.unescape(str(text or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"real", "true", "reliable", "1", "1.0"} or value.startswith("real"):
        return "REAL"
    if value in {"fake", "false", "unreliable", "0", "0.0"} or value.startswith("fake"):
        return "FAKE"
    raise ValueError(f"Unsupported label: {value!r}. Use REAL/FAKE or 1/0.")


def load_and_prepare_data(csv_path=DATASET_PATH):
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"news.csv is missing columns: {sorted(missing)}")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].map(normalize_label)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def train_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(
        stop_words=ENGLISH_STOPWORDS, ngram_range=(1, 2), max_df=0.95, min_df=2
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)
    labels = ["FAKE", "REAL"]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, pos_label="REAL"),
        "report": classification_report(y_test, predictions, labels=labels, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions, labels=labels),
        "samples": len(df),
    }
    return model, vectorizer, metrics


def predict_news(model, vectorizer, news_text):
    vector = vectorizer.transform([clean_text(news_text)])
    probabilities = model.predict_proba(vector)[0]
    prediction = model.predict(vector)[0]
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
    return {"real": ranked[:limit], "fake": list(reversed(ranked[-limit:]))}
