from __future__ import annotations

from dataclasses import dataclass


def interval_intersection_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    """
    两个时间区间的交集长度（毫秒）。

    A=[a0,a1], B=[b0,b1]，交集为 [max(a0,b0), min(a1,b1)]，
    若 min(a1,b1) <= max(a0,b0) 则无重叠，交集长度为 0。
    """
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0, end - start)


def interval_union_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    """
    两个时间区间的并集长度（毫秒）。

    union = len(A) + len(B) - len(intersection)
    """
    return max(0, (a1 - a0)) + max(0, (b1 - b0)) - interval_intersection_ms(a0, a1, b0, b1)


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    """
    IoU(Intersection over Union)：衡量两个时间区间重叠程度的比例，范围 [0,1]。

    - 1 表示完全重叠
    - 0 表示不重叠
    """
    u = interval_union_ms(a0, a1, b0, b1)
    if u <= 0:
        return 0.0
    return interval_intersection_ms(a0, a1, b0, b1) / u


@dataclass(frozen=True)
class Match:
    """
    一对匹配结果（ref 的一个段 ↔ hyp 的一个段）。

    这里的 index 是字幕中的段编号（不是列表下标）。
    """
    ref_index: int
    hyp_index: int
    iou: float
    ref_start_ms: int
    ref_end_ms: int
    hyp_start_ms: int
    hyp_end_ms: int


@dataclass(frozen=True)
class TimeSegment:
    """
    时间段（不关心文本内容，仅用于时间轴对齐）。

    index: 段编号
    start_ms/end_ms: 起止时间（毫秒）
    """
    index: int
    start_ms: int
    end_ms: int


def greedy_match_by_iou(ref: list[TimeSegment], hyp: list[TimeSegment], min_iou: float = 0.0) -> list[Match]:
    """
    基于 IoU 的贪心一对一匹配。

    目标：从所有候选配对中挑出一组不冲突的匹配（ref/hyp 每段最多使用一次），尽量让 IoU 大。

    做法：
    1) 枚举所有 (ref_i, hyp_j) 计算 IoU
    2) 过滤 IoU < min_iou 的候选
    3) 按 IoU 从大到小排序
    4) 依次取候选，若 ref_i 或 hyp_j 已被占用则跳过

    这不是全局最优算法，但实现简单、对“时间段重叠匹配”足够实用。
    """
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
        r = ref[i]
        h = hyp[j]
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


def alignment_pr(ref: list[TimeSegment], hyp: list[TimeSegment], min_iou: float = 0.3) -> tuple[float, float]:
    """
    在 IoU 阈值下计算 Precision / Recall（以“段匹配”为单位）。

    - precision = matched / hyp_segments
      生成字幕(hyp)里有多少段能找到靠谱的 ref 对应（少“乱切/多余段”）

    - recall = matched / ref_segments
      内挂字幕(ref)里有多少段被 hyp 覆盖（少“漏段”）
    """
    matches = greedy_match_by_iou(ref, hyp, min_iou=min_iou)
    precision = len(matches) / max(1, len(hyp))
    recall = len(matches) / max(1, len(ref))
    return precision, recall
