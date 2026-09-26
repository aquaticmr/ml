#!/usr/bin/env python
"""
Root entrypoint for 05_train_matcher.py.
Forwards execution to src/05_train_matcher.py.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    src_script = Path(__file__).resolve().parent / "src" / "05_train_matcher.py"
    runpy.run_path(str(src_script), run_name="__main__")
