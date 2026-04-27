"""
model_trainer.py — Trains the TF-IDF + Logistic Regression pipeline
and evaluates it on the test set.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.exception import SpamDetectorException
from src.logger import logger
from src.utils import clean_text, save_object


@dataclass
class ModelTrainerConfig:
    """File paths for saved model artefacts."""
    model_path: str = os.path.join("artifacts", "model.pkl")
    vectorizer_path: str = os.path.join("artifacts", "vectorizer.pkl")


class ModelTrainer:
    """Trains and evaluates the spam classification model."""

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate(self, train_path: str, test_path: str) -> float:
        """
        Train on train_path, evaluate on test_path, save artefacts.

        Args:
            train_path: Path to the training CSV.
            test_path:  Path to the test CSV.

        Returns:
            Test accuracy as a float.
        """
        logger.info("Starting model training …")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df["clean_text"] = train_df["text"].apply(clean_text)
            test_df["clean_text"] = test_df["text"].apply(clean_text)

            vectorizer = TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )
            X_train = vectorizer.fit_transform(train_df["clean_text"])
            X_test = vectorizer.transform(test_df["clean_text"])

            model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            model.fit(X_train, train_df["spam"])

            y_pred = model.predict(X_test)
            acc = accuracy_score(test_df["spam"], y_pred)

            logger.info("Test Accuracy: %.4f", acc)
            logger.info(
                "\n%s",
                classification_report(
                    test_df["spam"], y_pred, target_names=["ham", "spam"]
                ),
            )

            save_object(self.config.model_path, model)
            save_object(self.config.vectorizer_path, vectorizer)

            return acc

        except Exception as e:
            raise SpamDetectorException(e, sys)
