#!/usr/bin/env python3
from __future__ import annotations

import cProfile
import json
import pstats
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import ensure_results  # noqa: E402
from asr_learning.lib.report import build_report  # noqa: E402


def main() -> None:
    pr = cProfile.Profile()
    pr.enable()
    build_report()
    pr.disable()
    buf = StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    text = buf.getvalue()
    path = ensure_results() / "profile.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(json.dumps({"profile": str(path)}, indent=2))


if __name__ == "__main__":
    main()
