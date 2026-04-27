"""
test_app.py — Unit tests for the NLP Spam Detector.

Run all tests:
    pytest tests/

Run with verbose output:
    pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from src.utils import clean_text


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestCleanText:
    """Tests for the clean_text preprocessing utility."""

    def test_lowercases_text(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "'" not in result

    def test_collapses_whitespace(self):
        result = clean_text("hello    world")
        assert result == "hello world"

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_numbers(self):
        result = clean_text("You won $1000!")
        assert "1000" in result

    def test_strips_leading_trailing_spaces(self):
        assert clean_text("  hello  ") == "hello"


# ── Prediction pipeline tests ─────────────────────────────────────────────────

class TestPredictPipeline:
    """Tests for the prediction pipeline using mocked artefacts."""

    @patch("src.pipeline.predict_pipeline.load_object")
    def test_predict_returns_spam(self, mock_load):
        """Spam-like text should return label='spam'."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.05, 0.95]]

        # Mock vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()

        mock_load.side_effect = [mock_model, mock_vectorizer]

        from src.pipeline.predict_pipeline import PredictPipeline
        pipeline = PredictPipeline()
        result = pipeline.predict("Congratulations you won a free gift card!")

        assert result["label"] == "spam"
        assert result["spam_probability"] == 0.95
        assert result["confidence"] == 0.95

    @patch("src.pipeline.predict_pipeline.load_object")
    def test_predict_returns_ham(self, mock_load):
        """Normal text should return label='ham'."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.97, 0.03]]

        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()

        mock_load.side_effect = [mock_model, mock_vectorizer]

        from src.pipeline.predict_pipeline import PredictPipeline
        pipeline = PredictPipeline()
        result = pipeline.predict("Hey, are we still meeting tomorrow?")

        assert result["label"] == "ham"
        assert result["spam_probability"] == 0.03
        assert result["confidence"] == 0.97

    @patch("src.pipeline.predict_pipeline.load_object")
    def test_predict_output_keys(self, mock_load):
        """Result dict must always contain the three required keys."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.9, 0.1]]

        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()

        mock_load.side_effect = [mock_model, mock_vectorizer]

        from src.pipeline.predict_pipeline import PredictPipeline
        pipeline = PredictPipeline()
        result = pipeline.predict("test message")

        assert "label" in result
        assert "confidence" in result
        assert "spam_probability" in result


# ── Flask API tests ───────────────────────────────────────────────────────────

class TestFlaskAPI:
    """Tests for the Flask REST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        # Patch artefact loading so tests don't need model.pkl on disk
        with patch("src.pipeline.predict_pipeline.load_object") as mock_load, \
             patch("explain.shap_explanation") as mock_shap:

            mock_model = MagicMock()
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = [[0.1, 0.9]]
            mock_model.coef_ = [MagicMock()]

            mock_vectorizer = MagicMock()
            mock_vectorizer.transform.return_value = MagicMock()
            mock_vectorizer.get_feature_names_out.return_value = []

            mock_load.side_effect = [mock_model, mock_vectorizer]
            mock_shap.return_value = {
                "method": "shap",
                "top_spam_words": [],
                "top_ham_words": [],
                "base_value": -1.0,
            }

            import app as flask_app
            flask_app.app.config["TESTING"] = True
            with flask_app.app.test_client() as client:
                yield client

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"

    def test_predict_missing_text(self, client):
        response = client.post(
            "/predict",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_empty_text(self, client):
        response = client.post(
            "/predict",
            json={"text": ""},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_valid_request(self, client):
        response = client.post(
            "/predict",
            json={"text": "You have won a free prize click here"},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "label" in data
        assert "confidence" in data
        assert "explanation" in data
