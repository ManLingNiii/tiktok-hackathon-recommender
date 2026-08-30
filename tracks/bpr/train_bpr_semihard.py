"""BPR semi-hard experiment entry point."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_bpr_semihard import *  # noqa: F401,F403

if __name__ == "__main__":
    source = os.path.join(ROOT, "src", "baseline_bpr_semihard.py")
    with open(source, encoding="utf-8") as handle:
        code = compile(handle.read(), source, "exec")
    exec(code, {"__name__": "__main__", "__file__": source})
