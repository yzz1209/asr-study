#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.adaptive import apply_route, guess_route  # noqa: E402
from asr_learning.lib.hallucination import detect_hallucinations  # noqa: E402
from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE  # noqa: E402
from asr_learning.lib.srt import load_srt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    origin, asr = load_srt(args.origin), load_srt(args.asr)
    guess = guess_route(origin, asr)
    drop = {h.index for h in detect_hallucinations(asr, origin)}
    filtered = apply_route(asr, guess.route, drop)
    print(
        json.dumps(
            {
                "route": guess.route,
                "reason": guess.reason,
                "speech_density": guess.speech_density,
                "cues_in": len(asr),
                "cues_out": len(filtered),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
