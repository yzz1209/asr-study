from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .srt import SrtSegment, joined_text
from .text import normalize_for_cer
from .time_align import greedy_match_by_iou

# Bag of Hallucinations — Whisper-family Chinese subtitle junk + generic YouTube/B站 patterns.
BOH_PHRASES = (
    "谢谢观看",
    "请不吝点赞",
    "订阅",
    "转发",
    "打赏",
    "字幕由",
    "优酷",
    "腾讯视频",
    "www.",
    "http",
    "字幕志愿者",
    "未完待续",
)


@dataclass(frozen=True)
class HallucinationHit:
    index: int
    start_ms: int
    end_ms: int
    text: str
    reasons: list[str]
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


def ngram_repeat_score(text: str, n: int = 2) -> float:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < n * 3:
        return 0.0
    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)


def detect_hallucinations(
    hyp: list[SrtSegment],
    ref: list[SrtSegment] | None = None,
    *,
    min_iou: float = 0.15,
    repeat_threshold: float = 0.55,
    logprobs: dict[int, tuple[float, float]] | None = None,
    logprob_cut: float = -1.0,
    no_speech_cut: float = 0.6,
) -> list[HallucinationHit]:
    """
    PDF 任务 10: 多维度幻觉检测
      1. BoH 短语
      2. 与 origin/VAD 无重叠（交叉验证）
      3. n-gram 重复
      4. avg_logprob / no_speech_prob（Belle/Whisper 日志）
    """
    matched_hyp: set[int] = set()
    origin_blob = ""
    if ref:
        matched_hyp = {m.hyp_index for m in greedy_match_by_iou(ref, hyp, min_iou=min_iou)}
        origin_blob = normalize_for_cer(joined_text(ref), keep_punct=False)

    hits: list[HallucinationHit] = []
    for seg in hyp:
        reasons: list[str] = []
        compact = re.sub(r"\s+", "", seg.text)
        compact_norm = normalize_for_cer(seg.text, keep_punct=False)
        for phrase in BOH_PHRASES:
            if phrase.lower() in compact.lower():
                reasons.append(f"boh:{phrase}")
        if ref is not None and seg.index not in matched_hyp:
            if compact_norm and compact_norm in origin_blob:
                reasons.append("time_mismatch")
            else:
                reasons.append("no_origin_overlap")
        score = ngram_repeat_score(seg.text)
        if score >= repeat_threshold:
            reasons.append(f"ngram_repeat:{score:.2f}")
        avg_lp = ns = None
        if logprobs and seg.index in logprobs:
            avg_lp, ns = logprobs[seg.index]
            if avg_lp < logprob_cut:
                reasons.append(f"low_logprob:{avg_lp:.2f}")
            if ns >= no_speech_cut:
                reasons.append(f"no_speech:{ns:.2f}")
        if reasons:
            hits.append(
                HallucinationHit(
                    index=seg.index,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    reasons=reasons,
                    avg_logprob=avg_lp,
                    no_speech_prob=ns,
                )
            )
    return hits


def hits_to_dicts(hits: list[HallucinationHit]) -> list[dict]:
    return [asdict(h) for h in hits]
