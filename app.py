"""
app.py — Flask REST API for spam classification with explainability.

Endpoints
---------
POST /predict          — classify text, return prediction + SHAP explanation
POST /predict/explain  — same as above but let caller choose method & top_n
GET  /health           — liveness probe
"""

import sys
import os

from flask import Flask, jsonify, request, render_template

from src.logger import logger
from src.exception import SpamDetectorException
from explain import explain
from model import load_model, predict

app = Flask(__name__)

# ── Load artefacts once at startup ────────────────────────────────────────────

logger.info("Loading model artefacts …")
MODEL, VECTORIZER = load_model()
logger.info("Model ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the web interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — always returns 200 if the app is up."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Classify input text and return a SHAP explanation.

    Request body (JSON):
        { "text": "<email or message content>" }

    Response (JSON):
        {
            "text":             str,
            "label":            "spam" | "ham",
            "confidence":       float,
            "spam_probability": float,
            "explanation": {
                "method":         "shap",
                "top_spam_words": [{"word": str, "shap_value": float}, ...],
                "top_ham_words":  [{"word": str, "shap_value": float}, ...],
                "base_value":     float
            }
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "'text' field must not be empty."}), 400

    result = predict(text, model=MODEL, vectorizer=VECTORIZER)
    exp = explain(text, MODEL, VECTORIZER, method="shap", top_n=10)

    return jsonify(
        {
            "text": text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "spam_probability": round(result["spam_probability"], 4),
            "explanation": exp,
        }
    )


@app.route("/predict/explain", methods=["POST"])
def predict_explain_endpoint():
    """
    Extended endpoint — allows the caller to choose the explanation method.

    Request body (JSON):
        {
            "text":   "<email or message content>",   # required
            "method": "shap" | "feature_importance",  # optional, default "shap"
            "top_n":  int                              # optional, default 10
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "'text' field must not be empty."}), 400

    method = data.get("method", "shap")
    if method not in ("shap", "feature_importance"):
        return jsonify({"error": "method must be 'shap' or 'feature_importance'."}), 400

    top_n = int(data.get("top_n", 10))

    result = predict(text, model=MODEL, vectorizer=VECTORIZER)
    exp = explain(text, MODEL, VECTORIZER, method=method, top_n=top_n)

    return jsonify(
        {
            "text": text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "spam_probability": round(result["spam_probability"], 4),
            "explanation": exp,
        }
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
