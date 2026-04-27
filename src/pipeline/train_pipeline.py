"""
train_pipeline.py — Orchestrates the full training pipeline.

Two ways to run:
    python model.py                          (simple, recommended for beginners)
    python -m src.pipeline.train_pipeline    (as a module)
    python src/pipeline/train_pipeline.py    (as a script)
"""

import os
import sys

# Ensure project root is on sys.path when run as a script directly
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_ROOT))

from src.exception import SpamDetectorException  # noqa: E402
from src.logger import logger  # noqa: E402
from src.components.data_ingestion import DataIngestion  # noqa: E402
from src.components.model_trainer import ModelTrainer  # noqa: E402


class TrainPipeline:
    """Runs data ingestion → model training in sequence."""

    def run(self, source_path: str = "emails.csv") -> None:
        try:
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 50)

            # Step 1 — Data ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate(source_path)

            # Step 2 — Model training
            trainer = ModelTrainer()
            accuracy = trainer.initiate(train_path, test_path)

            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETE  |  Accuracy: %.4f", accuracy)
            logger.info("=" * 50)

        except Exception as e:
            raise SpamDetectorException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
