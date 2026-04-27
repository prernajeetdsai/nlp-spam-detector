"""
utils.py — Shared utility functions used across the project.
"""

import os
import pickle
import re
import sys

from src.exception import SpamDetectorException
from src.logger import logger


def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation/digits noise, collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string ready for TF-IDF vectorisation.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_object(file_path: str, obj) -> None:
    """
    Serialise any Python object to disk with pickle.

    Args:
        file_path: Destination path (directories are created if needed).
        obj:       Object to serialise.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved object → %s", file_path)
    except Exception as e:
        raise SpamDetectorException(e, sys)


def load_object(file_path: str):
    """
    Load a pickled object from disk.

    Args:
        file_path: Path to the .pkl file.

    Returns:
        The deserialised Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise SpamDetectorException(e, sys)
