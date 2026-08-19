#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE  # noqa: E402
from asr_learning.lib.srt import load_srt, subtitle_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "origin": subtitle_stats(load_srt(args.origin)),
                "asr": subtitle_stats(load_srt(args.asr)),
                "why": "生产 SRT 规范：空行分块、HH:MM:SS,mmm、CPS 过高会闪。origin 段多、asr 段少 = 漏段。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
