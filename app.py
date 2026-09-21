import os
import re
import sys
from pathlib import Path

import streamlit as st

from detector import DATASET_PATH, load_and_prepare_data, predict_news, train_model
from nlp.language import detect_language
from rag.verifier import verify_claim

st.set_page_config(page_title="TruthLens AI", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); color: #f8fafc; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .hero { background: linear-gradient(120deg, #1d4ed8, #7c3aed); padding: 1.4rem 1.6rem; border-radius: 18px; box-shadow: 0 18px 40px rgba(59, 130, 246, 0.18); }
    .hero h1 { color: white; margin-bottom: 0.2rem; }
    .hero p { color: #dbeafe; margin: 0; }
    .info-box { background: rgba(148, 163, 184, 0.15); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 16px; padding: 1rem; }
    .metric { background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 14px; padding: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### TruthLens")
    st.caption("Free multilingual fake/news claim verification system")

    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded:
        st.success(f"Loaded {uploaded.name}")

    st.write("Dataset format:")
    st.code("text,label\nThis is true,REAL\nThis is false,FAKE")

st.markdown(
    '<div class="hero"><h1>TruthLens</h1><p>Multilingual claim verification with evidence-aware reasoning for English, Hindi, and Hinglish.</p></div>',
    unsafe_allow_html=True,
)

st.warning("This tool is for decision support. It does not replace primary-source verification or fact-checking.")

uploaded_dataset = None
if uploaded:
    uploaded_dataset = uploaded

try:
    dataset_path = uploaded_dataset if uploaded_dataset else DATASET_PATH
    if dataset_path is not None and hasattr(dataset_path, "read"):
        df = load_and_prepare_data(dataset_path)
    else:
        df = load_and_prepare_data(dataset_path)

    model, vectorizer, metrics = train_model(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", metrics["samples"])
    col2.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col3.metric("Precision", f"{metrics['precision']:.2%}")
    col4.metric("Recall", f"{metrics['recall']:.2%}")
except Exception as exc:
    st.info("Training dataset missing or invalid. You can still use the claim verifier below.")
    model = None
    vectorizer = None
    metrics = None
    st.caption(f"Model status: {exc}")

st.markdown("## Check a claim")
user_text = st.text_area("Paste a headline, article snippet, question, or claim", height=180, placeholder="Example: PM Modi announced a new AI policy in Delhi today")

if st.button("Analyze claim", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a claim or news passage.")
    else:
        language = detect_language(user_text)
        st.write(f"Detected language: **{language.upper()}**")

        if model is not None and vectorizer is not None:
            try:
                result = predict_news(model, vectorizer, user_text)
                st.markdown("### Model verdict")
                if result["label"] == "REAL NEWS":
                    st.success(f"Prediction: {result['label']} | Confidence: {result['confidence']:.2%}")
                else:
                    st.error(f"Prediction: {result['label']} | Confidence: {result['confidence']:.2%}")
            except Exception as exc:
                st.warning(f"Model inference failed: {exc}")

        verification = verify_claim(user_text, max_sources=5)
        st.markdown("### Evidence-backed assessment")

        verdict_color = {"SUPPORTED": "success", "CONTRADICTED": "error", "UNVERIFIED": "warning"}
        verdict = verification["verdict"]
        stat = verdict_color.get(verdict, "info")
        getattr(st, stat)(f"Verdict: {verdict}")
        st.write(f"Confidence: {verification['confidence']:.2%}")
        st.write("Reason:")
        st.write(verification["reason"])

        if verification.get("evidence"):
            st.markdown("#### Sources")
            for item in verification["evidence"]:
                st.markdown(f"- [{item['title']}]({item['url']})")
                if item.get("snippet"):
                    st.caption(item["snippet"])

st.markdown("## Model evaluation details")
if metrics is not None:
    st.code(metrics["report"])
    st.write(metrics["confusion_matrix"])
else:
    st.caption("No trained model yet; upload a valid CSV or place news.csv in the repository root.")
