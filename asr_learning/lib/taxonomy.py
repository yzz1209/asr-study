from __future__ import annotations

from collections import Counter

from .metrics import cer
from .srt import SrtSegment, joined_text
from .text import char_ngrams, normalize_for_cer
from .time_align import greedy_match_by_iou


def substitution_pairs(ref_text: str, hyp_text: str, *, keep_punct: bool = False) -> list[tuple[str, str]]:
    result = cer(ref_text, hyp_text, keep_punct=keep_punct)
    pairs: list[tuple[str, str]] = []
    for op in result["ops"]:
        if op["op"] == "S" and op["ref"] and op["hyp"]:
            pairs.append((op["ref"], op["hyp"]))
    return pairs


def unmatched_origin_spans(ops: list[dict], min_len: int = 2) -> list[str]:
    spans: list[str] = []
    buf: list[str] = []
    for op in ops:
        if op["op"] == "D" and op["ref"]:
            buf.append(op["ref"])
        else:
            if len(buf) >= min_len:
                spans.append("".join(buf))
            buf = []
    if len(buf) >= min_len:
        spans.append("".join(buf))
    return spans


def _ngram_counter(text: str) -> Counter[str]:
    norm = normalize_for_cer(text, keep_punct=False)
    counts: Counter[str] = Counter()
    for n in (2, 3, 4):
        counts.update(char_ngrams(norm, n))
    return counts


FUNCTION_EDGE = set("的了是啊呀呢吧着过就吗哦呵嗯吁驾与和在把被")


def _ok_term(g: str) -> bool:
    if len(g) < 2 or len(g) > 4:
        return False
    if g[0] in FUNCTION_EDGE or g[-1] in FUNCTION_EDGE:
        return False
    return True


def hotword_candidates(origin: list[SrtSegment], asr: list[SrtSegment], limit: int = 200) -> list[str]:
    """Domain terms that origin uses and ASR drops or never says — the language bottleneck list."""
    ref_text = joined_text(origin)
    hyp_text = joined_text(asr)
    origin_ng = _ngram_counter(ref_text)
    hyp_ng = _ngram_counter(hyp_text)
    scored: list[tuple[int, int, str]] = []
    for gram, n in origin_ng.items():
        if n < 2 or not _ok_term(gram):
            continue
        if hyp_ng.get(gram, 0) > 0:
            continue
        scored.append((n, len(gram), gram))
    scored.sort(key=lambda x: (-x[1], -x[0], x[2]))

    matches = greedy_match_by_iou(origin, asr, min_iou=0.3)
    used = {m.ref_index for m in matches}
    for seg in origin:
        if seg.index in used:
            continue
        g = normalize_for_cer(seg.text, keep_punct=False)
        if _ok_term(g) and hyp_ng.get(g, 0) == 0:
            scored.append((3, len(g), g))
    scored.sort(key=lambda x: (-x[1], -x[0], x[2]))

    ordered: list[str] = []
    for _, _, g in scored:
        if any(g in kept for kept in ordered):
            continue
        ordered.append(g)
        if len(ordered) >= limit:
            break
    return ordered


def language_bottleneck_report(origin: list[SrtSegment], asr: list[SrtSegment]) -> dict:
    """
    Production lesson: CER 高往往不是声学模型崩了，而是「语言层」瓶颈——
    影视专有名词、古装玄幻词、人名地名。
    """
    ref = joined_text(origin)
    hyp = joined_text(asr)
    raw = cer(ref, hyp, keep_punct=True)
    stripped = cer(ref, hyp, keep_punct=False)
    pairs = substitution_pairs(ref, hyp, keep_punct=False)
    pair_counts = Counter(pairs).most_common(15)
    spans = unmatched_origin_spans(stripped["ops"], min_len=2)
    span_counts = Counter(spans).most_common(20)
    words = hotword_candidates(origin, asr)
    return {
        "cer_keep_punct": raw["cer"],
        "cer_strip_punct": stripped["cer"],
        "punct_delta": raw["cer"] - stripped["cer"],
        "substitutions": stripped["substitutions"],
        "insertions": stripped["insertions"],
        "deletions": stripped["deletions"],
        "top_substitutions": [{"ref": a, "hyp": b, "n": n} for (a, b), n in pair_counts],
        "deleted_span_count": len(span_counts),
        "deleted_spans_head": [{"span": s[:24], "n": n, "len": len(s)} for s, n in span_counts[:8]],
        "hotword_candidates": words,
        "note": (
            "deletions 主导 → 漏识别/漏段；"
            "origin 高频、asr 零次的 2-4 字就是热词；"
            "punct_delta 大 → 先别用带标点的 CER 做模型对比。"
        ),
    }

