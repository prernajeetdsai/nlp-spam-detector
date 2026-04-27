"""
logger.py — Centralised logging for the spam detector project.
Logs are written to both the console and a timestamped log file
inside the  logs/  directory.
"""

import logging
import os
from datetime import datetime

# ── Log file setup ────────────────────────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
)

# ── Logging config ────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Also log to console so you see output in the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(levelname)s: %(message)s")
)

logger = logging.getLogger("SpamDetector")
logger.addHandler(console_handler)
