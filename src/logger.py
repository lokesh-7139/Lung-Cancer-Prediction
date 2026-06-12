"""
Run-logging utility.

Captures all console output into timestamped log files inside ``logs/``.
Terminal output is preserved unchanged — the log file is a mirror.
TensorFlow/Keras verbose noise is filtered out of the log file.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR


class _TeeFilter:
    """Write to both a real file and the original stdout, filtering TF noise.

    Parameters
    ----------
    log_file:
        Open handle to the on-disk log.
    original_stdout:
        The real ``sys.stdout`` that was captured before redirection.
    """

    # Substrings that indicate TF/Keras internal noise.
    _NOISE_PATTERNS: tuple[str, ...] = (
        "tensorflow",
        "WARNING:tensorflow",
        "INFO:tensorflow",
        "streaming stdout",
        "streaming stderr",
    )

    def __init__(
        self, log_file: io.TextIOWrapper, original_stdout: io.TextIOWrapper
    ) -> None:
        self._log_file = log_file
        self._original_stdout = original_stdout

    # -- file-like interface ------------------------------------------------

    def write(self, text: str) -> None:
        self._original_stdout.write(text)
        self._original_stdout.flush()

        if not any(p in text.lower() for p in self._NOISE_PATTERNS):
            self._log_file.write(text)
            self._log_file.flush()

    def flush(self) -> None:
        self._original_stdout.flush()
        self._log_file.flush()


def start_run_logging() -> None:
    """Create a timestamped log file and redirect stdout/stderr.

    Call this once at the very beginning of ``main()``, before any
    ``print()``.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path = LOGS_DIR / f"run_{timestamp}.log"

    log_file = log_path.open("w", encoding="utf-8")
    tee = _TeeFilter(log_file, sys.stdout)

    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    print(f"Run log: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)