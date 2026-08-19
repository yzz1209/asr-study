from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median

from .srt import SrtSegment


def interval_intersection_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0, end - start)


def interval_union_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, (a1 - a0)) + max(0, (b1 - b0)) - interval_intersection_ms(a0, a1, b0, b1)


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    u = interval_union_ms(a0, a1, b0, b1)
    if u <= 0:
        return 0.0
    return interval_intersection_ms(a0, a1, b0, b1) / u


@dataclass(frozen=True)
class Match:
    ref_index: int
    hyp_index: int
    iou: float
    ref_start_ms: int
    ref_end_ms: int
    hyp_start_ms: int
    hyp_end_ms: int

    @property
    def start_offset_ms(self) -> int:
        return self.hyp_start_ms - self.ref_start_ms

    @property
    def end_offset_ms(self) -> int:
        return self.hyp_end_ms - self.ref_end_ms


def greedy_match_by_iou(ref: list[SrtSegment], hyp: list[SrtSegment], min_iou: float = 0.0) -> list[Match]:
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(ref):
        for j, h in enumerate(hyp):
            iou = interval_iou(r.start_ms, r.end_ms, h.start_ms, h.end_ms)
            if iou >= min_iou:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_r: set[int] = set()
    used_h: set[int] = set()
    matches: list[Match] = []
    for iou, i, j in candidates:
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
        r, h = ref[i], hyp[j]
        matches.append(
            Match(
                ref_index=r.index,
                hyp_index=h.index,
                iou=iou,
                ref_start_ms=r.start_ms,
                ref_end_ms=r.end_ms,
                hyp_start_ms=h.start_ms,
                hyp_end_ms=h.end_ms,
            )
        )
    return matches


def alignment_summary(ref: list[SrtSegment], hyp: list[SrtSegment], min_iou: float = 0.3) -> dict:
    matches = greedy_match_by_iou(ref, hyp, min_iou=min_iou)
    used_r = {m.ref_index for m in matches}
    used_h = {m.hyp_index for m in matches}
    precision = len(matches) / max(1, len(hyp))
    recall = len(matches) / max(1, len(ref))
    offsets = [m.start_offset_ms for m in matches]
    ious = [m.iou for m in matches]
    summary = {
        "precision": precision,
        "recall": recall,
        "ref_segments": len(ref),
        "hyp_segments": len(hyp),
        "matched": len(matches),
        "unmatched_ref": len(ref) - len(used_r),
        "unmatched_hyp": len(hyp) - len(used_h),
        "min_iou": min_iou,
    }
    if matches:
        summary.update(
            {
                "iou_mean": mean(ious),
                "iou_median": median(ious),
                "start_offset_ms_mean": mean(offsets),
                "start_offset_ms_median": median(offsets),
                "start_offset_ms_min": min(offsets),
                "start_offset_ms_max": max(offsets),
            }
        )
    return summary


def diagnose_alignment(ref: list[SrtSegment], hyp: list[SrtSegment], min_iou: float = 0.3) -> dict:
    """Map alignment numbers onto production failure modes: 漏段 / 乱切 / 固定偏移 / 漂移."""
    matches = greedy_match_by_iou(ref, hyp, min_iou=min_iou)
    summary = alignment_summary(ref, hyp, min_iou=min_iou)
    modes: list[str] = []
    if summary["recall"] < 0.7:
        modes.append("漏段：origin 大量段没被 ASR 覆盖（VAD 过猛 / 模型吞词 / 切段粒度差）")
    if summary["precision"] < 0.8:
        modes.append("乱切或多余段：ASR 段找不到 origin 对应（幻觉 / 背景音当说话 / 切太碎）")
    if matches:
        mean_off = abs(float(summary["start_offset_ms_mean"]))
        spread = float(summary["start_offset_ms_max"]) - float(summary["start_offset_ms_min"])
        if mean_off > 200 and spread < 800:
            modes.append("固定偏移：整体早/晚，优先查解码器时间基、VAD pad、转封装")
        if spread > 1500:
            modes.append("漂移：偏移随时间变大，优先查分段累计误差 / 变声变速 / FA 失败")
    if not modes:
        modes.append("时间轴整体可用，继续看 CER 和专有名词错误")
    return {
        **summary,
        "failure_modes": modes,
        "worst_by_iou": [asdict(m) for m in sorted(matches, key=lambda x: x.iou)[:5]],
        "unmatched_ref_indexes": [s.index for s in ref if s.index not in {m.ref_index for m in matches}][:20],
        "unmatched_hyp_indexes": [s.index for s in hyp if s.index not in {m.hyp_index for m in matches}][:20],
    }
