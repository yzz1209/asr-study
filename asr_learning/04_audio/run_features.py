#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.features import hz_to_mel, mel_filterbank, spec_augment_mask  # noqa: E402


def main() -> None:
    fb = mel_filterbank(n_fft=512, sr=16000, n_mels=40)
    nonzero = sum(1 for row in fb for x in row if x > 0)
    print(
        json.dumps(
            {
                "why_16k": "电话 8kHz 会切掉摩擦音和辅音细节，影视对白 CER 会掉；生产 PCM 固定 16k/16bit mono。",
                "hz_1000_in_mel": hz_to_mel(1000),
                "filterbank_rows": len(fb),
                "nonzero_weights": nonzero,
                "specaug_drop_ratio": sum(1 for r in spec_augment_mask(20, 40, seed=1) for x in r if x == 0) / (20 * 40),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
