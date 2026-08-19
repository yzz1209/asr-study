#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE, ensure_results  # noqa: E402
from asr_learning.lib.srt import load_srt  # noqa: E402
from asr_learning.lib.text_align import scheme_a_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="方案A / A2：Belle 文本贴 Paraformer 时间戳")
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    report = scheme_a_report(load_srt(args.origin), load_srt(args.asr))
    path = ensure_results() / "scheme_a.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
