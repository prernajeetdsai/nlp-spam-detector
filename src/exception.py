"""
exception.py — Custom exception class for the spam detector.
Captures the file name and line number where the error occurred.
"""

import sys


def get_error_message(error, error_detail: sys) -> str:
    """Build a detailed error string with file and line info."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error in script: [{file_name}] "
        f"at line [{line_number}] — {str(error)}"
    )


class SpamDetectorException(Exception):
    """
    Custom exception for the NLP Spam Detector project.

    Usage:
        try:
            ...
        except Exception as e:
            raise SpamDetectorException(e, sys)
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_message(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
