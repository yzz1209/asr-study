from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import inf
from typing import Callable, Sequence, TypeVar

from .srt import SrtSegment

T = TypeVar("T")


@dataclass(frozen=True)
class DtwResult:
    cost: float
    path: list[tuple[int, int]]


def dtw_align(seq1: Sequence[T], seq2: Sequence[T], dist: Callable[[T, T], float] | None = None) -> DtwResult:
    if dist is None:
        dist = lambda a, b: abs(float(a) - float(b))  # type: ignore[misc, arg-type]
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return DtwResult(cost=inf if n != m else 0.0, path=[])
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(dist(seq1[i - 1], seq2[j - 1]))
            prevs = (
                (dp[i - 1][j], (i - 1, j)),
                (dp[i][j - 1], (i, j - 1)),
                (dp[i - 1][j - 1], (i - 1, j - 1)),
            )
            best_prev_cost, best_prev = min(prevs, key=lambda x: x[0])
            dp[i][j] = best_prev_cost + cost
            bt[i][j] = best_prev
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = bt[i][j]
        if prev is None:
            break
        i, j = prev
    path.reverse()
    return DtwResult(cost=dp[n][m], path=path)


@dataclass(frozen=True)
class CharStamp:
    ch: str
    start_ms: int
    end_ms: int


def explode_chars(segments: list[SrtSegment]) -> list[CharStamp]:
    """Linearly assign timestamps to characters inside each cue — Paraformer-like word stamps."""
    out: list[CharStamp] = []
    for seg in segments:
        chars = [c for c in seg.text if not c.isspace()]
        if not chars:
            continue
        dur = max(1, seg.end_ms - seg.start_ms)
        slot = dur / len(chars)
        for k, ch in enumerate(chars):
            start = int(seg.start_ms + k * slot)
            end = int(seg.start_ms + (k + 1) * slot)
            out.append(CharStamp(ch=ch, start_ms=start, end_ms=end))
    return out


@dataclass(frozen=True)
class TextAlignResult:
    """方案 A：把「较好文本」贴到「带时间戳的字/词序列」上。"""

    aligned: list[CharStamp]
    mean_err_ms: float | None
    matched_chars: int
    coverage: float


def align_text_to_stamped_chars(
    belle_text: str,
    pf_chars: list[CharStamp],
) -> TextAlignResult:
    """
    Production 方案 A (PDF 任务 8): align Belle text onto Paraformer word timestamps
    with string matching, not WhisperX.

    Here we use SequenceMatcher (Ratcliff/Obershelp; related to LCS thinking).
    """
    belle = [c for c in belle_text if not c.isspace()]
    pf = list(pf_chars)
    matcher = SequenceMatcher(a=belle, b=[c.ch for c in pf], autojunk=False)
    aligned: list[CharStamp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned.extend(pf[j1:j2])
        elif tag == "replace":
            # Keep Paraformer time, take Belle characters where lengths allow.
            span = pf[j1:j2]
            for k, ch in enumerate(belle[i1:i2]):
                if not span:
                    continue
                src = span[min(k, len(span) - 1)]
                aligned.append(CharStamp(ch=ch, start_ms=src.start_ms, end_ms=src.end_ms))
        elif tag == "insert":
            continue
        elif tag == "delete":
            continue
    return TextAlignResult(aligned=aligned, mean_err_ms=None, matched_chars=len(aligned), coverage=0.0)


def mean_err_vs_origin(aligned: list[CharStamp], origin: list[SrtSegment]) -> float | None:
    """Mean absolute start error of aligned chars vs the origin cue that contains that char index-wise.

    Simpler production metric from the PDF: MeanErr < 800ms on a test set.
    We compare each aligned char start to the nearest origin segment start.
    """
    if not aligned or not origin:
        return None
    errors: list[int] = []
    for ch in aligned:
        nearest = min(origin, key=lambda s: min(abs(ch.start_ms - s.start_ms), abs(ch.end_ms - s.end_ms)))
        errors.append(min(abs(ch.start_ms - nearest.start_ms), abs(ch.end_ms - nearest.end_ms)))
    return sum(errors) / len(errors)


def scheme_a_report(origin: list[SrtSegment], asr: list[SrtSegment]) -> dict:
    """
    Miniature of 方案 A / A2 vertical path:
    - origin text ≈ Belle-quality transcript (more complete, fewer phonetic errors)
    - asr cues ≈ Paraformer-like timestamps (what the time-accurate branch produced)
    Glue them with string matching and report MeanErr.
    """
    belle_text = "".join(s.text for s in origin)
    pf_chars = explode_chars(asr)
    result = align_text_to_stamped_chars(belle_text, pf_chars)
    mean_err = mean_err_vs_origin(result.aligned, origin)
    coverage = len(result.aligned) / max(1, len([c for c in belle_text if not c.isspace()]))
    return {
        "scheme": "A2-text-forced-alignment",
        "belle_chars": len([c for c in belle_text if not c.isspace()]),
        "paraformer_stamped_chars": len(pf_chars),
        "aligned_chars": len(result.aligned),
        "coverage": coverage,
        "mean_err_ms": mean_err,
        "pass_mean_err_800ms": (mean_err is not None and mean_err < 800),
    }
