# Fake News Detector

A simple machine-learning project that classifies news articles as real or fake using TF-IDF features and logistic regression.

## What this project does

- Loads a dataset from `news.csv`
- Cleans and preprocesses the text
- Converts article text to TF-IDF vectors
- Trains a Logistic Regression model
- Evaluates the model on a test set
- Predicts whether a new article is likely real or fake

## Run it

1. Make sure Python is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your dataset file named `news.csv` in the project folder.
4. Run the script:

```bash
python Code
```

You can also pass a sentence directly:

```bash
python Code "This is a headline about a local city event and a new public policy update."
```

## Notes

- The dataset should contain at least two columns: `text` and `label`.
- Labels may be `REAL` / `FAKE` or `1` / `0`.
- The script uses a logistic regression model and is meant as a simple text-classification baseline.
