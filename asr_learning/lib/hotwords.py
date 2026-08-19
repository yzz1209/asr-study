from __future__ import annotations

from .taxonomy import language_bottleneck_report
from .srt import load_srt


def extract_hotwords(origin_path: str, asr_path: str, limit: int = 200) -> list[str]:
    origin = load_srt(origin_path)
    asr = load_srt(asr_path)
    report = language_bottleneck_report(origin, asr)
    words = report["hotword_candidates"][:limit]
    # Also keep frequent origin-only names that look like domain terms (len 2-6).
    return words
