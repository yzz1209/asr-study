#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.hotwords import extract_hotwords  # noqa: E402
from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE, ensure_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    words = extract_hotwords(args.origin, args.asr, limit=args.limit)
    path = ensure_results() / "hotwords.txt"
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(words), "path": str(path), "head": words[:30]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
