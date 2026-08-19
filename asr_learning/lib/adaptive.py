from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .srt import SrtSegment, subtitle_stats


Route = Literal["belle_only", "paraformer_plus_fa", "belle_aggressive_filter"]


@dataclass(frozen=True)
class AudioGuess:
    """Without raw audio we infer a routing guess from subtitle density — same decision surface as PDF 任务14."""

    speech_density: float
    cue_cps_mean: float
    long_gap_ratio: float
    route: Route
    reason: str


def guess_route(origin: list[SrtSegment] | None, hyp: list[SrtSegment]) -> AudioGuess:
    stats = subtitle_stats(hyp)
    span = max(1, stats.get("span_ms", 1))
    speech_ms = sum(max(0, s.end_ms - s.start_ms) for s in hyp)
    density = speech_ms / span
    cps = float(stats.get("cps_mean") or 0)
    gaps = []
    for i in range(1, len(hyp)):
        gaps.append(hyp[i].start_ms - hyp[i - 1].end_ms)
    long_gap_ratio = (sum(1 for g in gaps if g > 3000) / max(1, len(gaps))) if gaps else 0.0

    # Music-heavy film tracks often have sparse speech + high cps bursts.
    if density < 0.25 and long_gap_ratio > 0.2:
        route: Route = "belle_aggressive_filter"
        reason = "稀疏对白+长静音，偏影视配乐轨：Belle + 强幻觉过滤"
    elif density > 0.55 and cps >= 6:
        route = "belle_only"
        reason = "对白密、节奏稳，PDF 里的 clean speech：跳过 FA，只用 Belle"
    else:
        route = "paraformer_plus_fa"
        reason = "中等噪声/切段不稳：Paraformer 拿时间戳 + 方案A 把 Belle 文本贴上去"

    return AudioGuess(
        speech_density=density,
        cue_cps_mean=cps,
        long_gap_ratio=long_gap_ratio,
        route=route,
        reason=reason,
    )


def apply_route(hyp: list[SrtSegment], route: Route, drop_indexes: set[int] | None = None) -> list[SrtSegment]:
    if route != "belle_aggressive_filter" or not drop_indexes:
        return list(hyp)
    return [s for s in hyp if s.index not in drop_indexes]
