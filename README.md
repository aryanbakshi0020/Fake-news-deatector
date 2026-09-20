# TruthLens — AI News Credibility Analyzer

TruthLens is a polished Streamlit application that screens news text as likely REAL or FAKE using TF-IDF features and balanced Logistic Regression. It includes confidence scores, evaluation metrics, dataset upload, and interpretable term-level signals.

> A prediction is a statistical signal, not proof that an article is true or false. Verify important claims with primary sources and reputable fact-checkers.

## Dataset format

The default `news.csv` belongs in the repository root. You can also upload another CSV directly from the app sidebar. Every dataset must contain `text` and `label` columns:

```csv
text,label
"Your article text",REAL
"Another article",FAKE
```

Labels can be `REAL`/`FAKE`, `TRUE`/`FALSE`, or `1`/`0`. Uploaded datasets should contain at least 8 rows and both classes.

## Run locally

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect GitHub.
3. Select `aryanbakshi0020/Fake-news-deatector`.
4. Set the main file to `app.py` and deploy.

## Portfolio highlights

- Premium responsive interface with custom visual design
- Default dataset or instant CSV upload and retraining
- Stratified evaluation with accuracy, F1 score, and confusion matrix
- Confidence scores and explainable TF-IDF feature signals
- Responsible-AI limitation notice
