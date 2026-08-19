#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.features import spec_augment_mask  # noqa: E402


def main() -> None:
    mask = spec_augment_mask(n_frames=32, n_mels=16, time_mask=6, freq_mask=3, seed=3)
    dropped = sum(1 for row in mask for x in row if x == 0)
    total = len(mask) * len(mask[0])
    print(json.dumps({"frames": 32, "mels": 16, "dropped_ratio": dropped / total, "mask_row0": mask[0]}, indent=2))


if __name__ == "__main__":
    main()
