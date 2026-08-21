#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.ctc import ctc_forced_alignment, ctc_greedy_decode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CTC greedy + 迷你 forced alignment（Paraformer/WhisperX 直觉）")
    parser.add_argument("--text", default="灵相师")
    parser.add_argument(
        "--frames",
        default="<blank>,灵,灵,相,<blank>,师,师,<blank>",
        help="comma-separated CTC frame tokens",
    )
    args = parser.parse_args()
    frames = [tok.strip() for tok in args.frames.split(",") if tok.strip()]
    decoded = ctc_greedy_decode(frames)
    fa = ctc_forced_alignment(args.text, frames)
    print(json.dumps({"decoded": decoded, "forced_alignment": fa, "frames": frames}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
