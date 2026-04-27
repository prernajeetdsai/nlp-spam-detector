"""
model.py — Training and loading logic for the spam classifier.
Run directly to train:  python model.py
"""

import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.logger import logger
from src.exception import SpamDetectorException
from src.utils import clean_text, save_object, load_object

# ── Artefact paths (saved inside artifacts/ folder) ───────────────────────────

MODEL_PATH = "artifacts/model.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str = "emails.csv") -> None:
    """Load data, train TF-IDF + Logistic Regression, persist artefacts."""
    try:
        logger.info("Loading dataset from %s", data_path)
        df = pd.read_csv(data_path).dropna()
        df["clean_text"] = df["text"].apply(clean_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"],
            df["spam"],
            test_size=0.2,
            random_state=42,
            stratify=df["spam"],
        )

        logger.info("Fitting TF-IDF vectorizer …")
        vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info("Training Logistic Regression …")
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        logger.info("Accuracy: %.4f", accuracy_score(y_test, y_pred))
        logger.info(
            "\n%s",
            classification_report(y_test, y_pred, target_names=["ham", "spam"]),
        )

        # Save using shared utility from src/utils.py
        save_object(MODEL_PATH, model)
        save_object(VECTORIZER_PATH, vectorizer)

    except Exception as e:
        raise SpamDetectorException(e, sys)


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_model():
    """Return (model, vectorizer) from artifacts/ folder."""
    try:
        model = load_object(MODEL_PATH)
        vectorizer = load_object(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        raise SpamDetectorException(e, sys)


def predict(text: str, model=None, vectorizer=None) -> dict:
    """
    Classify a single piece of text.

    Returns:
        {
            "label":            "spam" | "ham",
            "confidence":       float,
            "spam_probability": float,
        }
    """
    try:
        if model is None or vectorizer is None:
            model, vectorizer = load_model()

        cleaned = clean_text(text)
        tfidf = vectorizer.transform([cleaned])
        label_idx = model.predict(tfidf)[0]
        proba = model.predict_proba(tfidf)[0]

        return {
            "label": "spam" if label_idx == 1 else "ham",
            "confidence": float(proba[label_idx]),
            "spam_probability": float(proba[1]),
        }
    except Exception as e:
        raise SpamDetectorException(e, sys)


if __name__ == "__main__":
    train()
