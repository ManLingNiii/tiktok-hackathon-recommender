"""Compatibility re-export; the model logic remains in src/baseline_bpr.py."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_bpr import FM  # noqa: F401
