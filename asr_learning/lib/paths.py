from __future__ import annotations

from pathlib import Path

ASR_LEARNING = Path(__file__).resolve().parents[1]
REPO_ROOT = ASR_LEARNING.parent
DATA_SRT = ASR_LEARNING / "data" / "srt"
DATA_AUDIO = ASR_LEARNING / "data" / "raw_audio"
RESULTS = ASR_LEARNING / "results"

ORIGIN_SAMPLE = DATA_SRT / "origin.sample.srt"
ASR_SAMPLE = DATA_SRT / "asr.sample.srt"


def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS
