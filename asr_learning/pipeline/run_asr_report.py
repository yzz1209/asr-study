#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE, ensure_results  # noqa: E402
from asr_learning.lib.report import build_report, write_html, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="生成对照真实管线的 ASR 字幕质量报告")
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    ensure_results()
    report = build_report(args.origin, args.asr)
    json_path = write_json(report)
    html_path = write_html(report)
    print(json.dumps({"json": str(json_path), "html": str(html_path), "readout": report["readout"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
