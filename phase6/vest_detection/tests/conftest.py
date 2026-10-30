"""
conftest.py
-----------
Pytest configuration for vest_detection module.
Adds src/ to sys.path so tests can import vest_detector seamlessly.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
