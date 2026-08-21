#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.hallucination import detect_hallucinations, hits_to_dicts  # noqa: E402
from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE  # noqa: E402
from asr_learning.lib.srt import load_srt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    hits = detect_hallucinations(load_srt(args.asr), load_srt(args.origin))
    print(json.dumps({"count": len(hits), "hits": hits_to_dicts(hits)[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
