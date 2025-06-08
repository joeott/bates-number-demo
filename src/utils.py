import os
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path: Path):
    """Creates a directory if it doesn't exist."""
    dir_path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    # Basic sanitization, can be expanded
    return "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename)