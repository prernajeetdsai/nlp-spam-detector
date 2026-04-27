"""
predict_pipeline.py — Loads saved artefacts and classifies new text.
Used by app.py to serve predictions via the REST API.
"""

import os
import sys

from src.exception import SpamDetectorException
from src.logger import logger
from src.utils import clean_text, load_object

MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "vectorizer.pkl")


class PredictPipeline:
    """Loads model + vectorizer and exposes a predict method."""

    def __init__(self):
        try:
            self.model = load_object(MODEL_PATH)
            self.vectorizer = load_object(VECTORIZER_PATH)
            logger.info("PredictPipeline: artefacts loaded.")
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def predict(self, text: str) -> dict:
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
            cleaned = clean_text(text)
            tfidf = self.vectorizer.transform([cleaned])
            label_idx = self.model.predict(tfidf)[0]
            proba = self.model.predict_proba(tfidf)[0]

            return {
                "label": "spam" if label_idx == 1 else "ham",
                "confidence": round(float(proba[label_idx]), 4),
                "spam_probability": round(float(proba[1]), 4),
            }
        except Exception as e:
            raise SpamDetectorException(e, sys)
