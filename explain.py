"""
explain.py — Explainability layer for the spam classifier.

Two methods are provided:
  1. feature_importance  — fast, coefficient-based top-N words (always available)
  2. shap_explanation    — SHAP LinearExplainer values for the input tokens
"""

import sys
from typing import Optional

import numpy as np
import shap

from src.logger import logger
from src.exception import SpamDetectorException
from src.utils import clean_text

# ── Feature-importance explanation ───────────────────────────────────────────

def feature_importance_explanation(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> dict:
    """
    Return the top-N words most responsible for the prediction by multiplying
    the TF-IDF score of each token in the input by the model's log-odds
    coefficient for the spam class.

    Returns:
        {
            "method": "feature_importance",
            "top_spam_words":  [{"word": str, "score": float}, ...],
            "top_ham_words":   [{"word": str, "score": float}, ...],
        }
    """
    cleaned = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Coefficients for the spam class (index 1 in binary LR)
    coefs = model.coef_[0]  # shape (n_features,)

    # Only consider features present in this document
    nonzero_idx = tfidf_vec.nonzero()[1]
    if len(nonzero_idx) == 0:
        return {
            "method": "feature_importance",
            "top_spam_words": [],
            "top_ham_words": [],
        }

    tfidf_scores = np.array(tfidf_vec[0, nonzero_idx].todense()).flatten()
    word_scores = tfidf_scores * coefs[nonzero_idx]
    words = feature_names[nonzero_idx]

    # Sort descending for spam words, ascending for ham words
    sorted_idx = np.argsort(word_scores)[::-1]
    sorted_words = words[sorted_idx]
    sorted_scores = word_scores[sorted_idx]

    spam_words = [
        {"word": w, "score": round(float(s), 4)}
        for w, s in zip(sorted_words[:top_n], sorted_scores[:top_n])
        if s > 0
    ]
    ham_words = [
        {"word": w, "score": round(float(abs(s)), 4)}
        for w, s in zip(sorted_words[-top_n:][::-1], sorted_scores[-top_n:][::-1])
        if s < 0
    ]

    return {
        "method": "feature_importance",
        "top_spam_words": spam_words,
        "top_ham_words": ham_words,
    }


# ── SHAP explanation ──────────────────────────────────────────────────────────

# Module-level cache so the explainer is built only once per process
_shap_explainer: Optional[shap.LinearExplainer] = None


def _get_shap_explainer(model, vectorizer):
    global _shap_explainer
    if _shap_explainer is None:
        logger.info("Building SHAP LinearExplainer (one-time cost) …")
        # masker = Independent → treats each feature independently (fast for sparse)
        masker = shap.maskers.Independent(
            np.zeros((1, len(vectorizer.get_feature_names_out())))
        )
        _shap_explainer = shap.LinearExplainer(model, masker=masker)
    return _shap_explainer


def shap_explanation(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> dict:
    """
    Compute SHAP values for the spam class and return the top contributing
    words (positive = push toward spam, negative = push toward ham).

    Returns:
        {
            "method": "shap",
            "top_spam_words":  [{"word": str, "shap_value": float}, ...],
            "top_ham_words":   [{"word": str, "shap_value": float}, ...],
            "base_value": float,
        }
    """
    cleaned = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    explainer = _get_shap_explainer(model, vectorizer)
    shap_values = explainer.shap_values(tfidf_vec)

    # shap_values shape: (n_classes, n_samples, n_features) for multi-output LR
    # or (n_samples, n_features) for binary — handle both
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # spam class
    else:
        sv = shap_values[0]

    nonzero_idx = np.where(sv != 0)[0]
    if len(nonzero_idx) == 0:
        return {
            "method": "shap",
            "top_spam_words": [],
            "top_ham_words": [],
            "base_value": float(explainer.expected_value if np.isscalar(explainer.expected_value)
                                else explainer.expected_value[1]),
        }

    words = feature_names[nonzero_idx]
    values = sv[nonzero_idx]
    sorted_idx = np.argsort(values)[::-1]

    top_spam = [
        {"word": words[i], "shap_value": round(float(values[i]), 4)}
        for i in sorted_idx[:top_n]
        if values[i] > 0
    ]
    top_ham = [
        {"word": words[i], "shap_value": round(float(values[i]), 4)}
        for i in sorted_idx[-top_n:][::-1]
        if values[i] < 0
    ]

    base = explainer.expected_value
    base_val = float(base if np.isscalar(base) else base[1])

    return {
        "method": "shap",
        "top_spam_words": top_spam,
        "top_ham_words": top_ham,
        "base_value": round(base_val, 4),
    }


# ── Public interface ──────────────────────────────────────────────────────────

def explain(
    text: str,
    model,
    vectorizer,
    method: str = "shap",
    top_n: int = 10,
) -> dict:
    """
    Unified explanation entry-point.

    Args:
        text:        Raw input text.
        model:       Trained LogisticRegression.
        vectorizer:  Fitted TfidfVectorizer.
        method:      "shap" (default) or "feature_importance".
        top_n:       Number of top words to return per direction.

    Returns:
        Explanation dict (see individual functions for schema).
    """
    if method == "shap":
        return shap_explanation(text, model, vectorizer, top_n=top_n)
    elif method == "feature_importance":
        return feature_importance_explanation(text, model, vectorizer, top_n=top_n)
    else:
        raise ValueError(f"Unknown explanation method '{method}'. Use 'shap' or 'feature_importance'.")
