# Fake News Detector v2

A free, Colab-friendly multilingual fake-news detection and verification system for English, Hindi, and Hinglish.

This project is designed for research and portfolio use in Google Colab with low CPU/RAM usage and no paid API requirement.

Important note:
- Accuracy, precision, recall, specificity, and F1 can never exceed 100% because they are percentages.
- We use statistically valid evaluation and proper train/validation/test separation to avoid overfitting and underfitting.
- The system also supports evidence-based claim checking using web search results and source links.

## Project structure

```text
fake-news-detector/
├── app.py
├── detector.py
├── requirements.txt
├── .gitignore
├── README.md
├── notebooks/
│   └── 01_dataset_refinement_training.ipynb
├── data/
│   ├── raw/
│   ├── refined/
│   └── external/
├── models/
│   ├── baseline/
│   └── transformer/
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   └── verifier.py
├── nlp/
│   ├── __init__.py
│   └── language.py
└── tests/
    └── sample_claims.txt
```

## Recommended Kaggle dataset

Use:
- `maulishkasrivastava/multilingual-fake-news-mbert`
- or the exact dataset name: `Multilingual Fake News Detection Dataset for mBERT`

This dataset is the best starting point because it combines English, Hindi, and Hinglish/code-mixed text patterns.

Additional Hindi-only datasets can be added later to improve robustness.

## What this project contains

- Multilingual language detection for English/Hindi/Hinglish
- CSV-based dataset refinement and label normalization
- Baseline TF-IDF + Logistic Regression model
- Transformer-ready notebook for Colab
- Web evidence retrieval and claim checking via DuckDuckGo
- Clean, modular code with error handling and file-based organization
- CPU-friendly design for free Google Colab usage

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Colab setup

1. Upload this project to Google Drive or GitHub.
2. Open Colab.
3. Install requirements:

```bash
!pip install -r requirements.txt
```

4. Run the notebook:

```python
!jupyter nbconvert --to notebook --execute notebooks/01_dataset_refinement_training.ipynb
```

## Notebook workflow

The notebook creates:
- `data/raw/` from Kaggle or local source files
- `data/refined/` using cleaning, de-duplication, normalization, and label enforcement
- `models/baseline/` for logistic regression and TF-IDF artifacts
- `models/transformer/` for downloaded model checkpoints

## Model quality expectations

Use proper statistics and probability-based validation:
- accuracy
- precision
- recall
- specificity
- F1-score
- confusion matrix
- balanced accuracy
- calibration and cross-validation

A valid model should aim for strong metrics, but not claim more than 100%.

## Evidence-based verification

The app supports claim checking:
- detect language
- form a search query
- fetch current web evidence
- rank the most relevant URLs
- provide source links and verdict reasoning

This helps answer questions like:
- Is this claim fake or supported?
- Is this a real statement or misinformation?
- What evidence is available from trusted sources?

## Responsible use

This project is a decision-support tool, not a final truth oracle.

Always cross-check critical information with:
- official government sources
- reputable news organizations
- fact-checking organizations
- primary references

## Future upgrades

- DistilBERT / XLM-R multilingual model fine-tuning
- sentence embeddings for semantic retrieval
- better Hindi and Hinglish normalization
- REST API deployment
- Docker packaging
- stronger RAG ranking and source filtering
