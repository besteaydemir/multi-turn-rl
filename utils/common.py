"""Common utility functions."""

from datetime import datetime


def timestamp_str():
    """Return current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
