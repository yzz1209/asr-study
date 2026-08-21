#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import DATA_AUDIO  # noqa: E402
from asr_learning.lib.vad import energy_vad_wav, write_sine_wav  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_AUDIO / "demo_speech.wav"))
    args = parser.parse_args()
    DATA_AUDIO.mkdir(parents=True, exist_ok=True)
    write_sine_wav(
        args.out,
        events=[
            (0.40, 1.10, 220.0),
            (1.60, 2.40, 330.0),
        ],
    )
    segs = energy_vad_wav(args.out)
    print(json.dumps({"wav": args.out, "segments": [s.__dict__ for s in segs]}, indent=2))


if __name__ == "__main__":
    main()
