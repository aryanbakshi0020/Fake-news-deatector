# TruthLens — AI News Credibility Analyzer

TruthLens is a portfolio-ready Streamlit application that uses TF-IDF features and balanced Logistic Regression to screen news text as likely REAL or FAKE. It includes confidence scores, model metrics, and interpretable term-level signals.

> This model detects patterns learned from the supplied dataset; it does not prove whether a claim is true. Always verify consequential information with reputable sources.

## Dataset

Place `news.csv` in the repository root. It must contain:

```csv
text,label
"Your article text",REAL
"Another article",FAKE
```

Labels can be `REAL`/`FAKE` or `1`/`0`.

## Run locally

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The original command-line version remains available with `python Code`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including `app.py`, `detector.py`, `requirements.txt`, and `news.csv` (or provide the dataset through a private data workflow).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect GitHub.
3. Select `aryanbakshi0020/Fake-news-deatector`.
4. Set the main file to `app.py` and deploy.

## Portfolio highlights

- Reproducible preprocessing and stratified evaluation
- Confidence scores and F1/accuracy metrics
- Explainable TF-IDF feature signals
- Responsive web UI with graceful missing-data errors
- Clear limitation notice for responsible ML use
