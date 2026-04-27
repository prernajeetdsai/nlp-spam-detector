"""
data_ingestion.py — Loads the raw CSV and splits it into train / test sets.
"""

import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import SpamDetectorException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    """File paths used by the data ingestion step."""
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """Reads the raw email CSV and produces train/test split files."""

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate(self, source_path: str = "emails.csv"):
        """
        Load data from source_path, split, and save to artifacts/.

        Args:
            source_path: Path to the raw emails CSV.

        Returns:
            Tuple of (train_path, test_path).
        """
        logger.info("Starting data ingestion from: %s", source_path)
        try:
            df = pd.read_csv(source_path)
            logger.info("Dataset loaded — shape: %s", df.shape)

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["spam"]
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(
                "Train size: %d | Test size: %d", len(train_df), len(test_df)
            )
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise SpamDetectorException(e, sys)
