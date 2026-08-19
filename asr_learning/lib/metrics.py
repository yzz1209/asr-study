from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, TypeVar

from .text import normalize_for_cer

T = TypeVar("T")


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions


@dataclass(frozen=True)
class EditOp:
    op: str  # M / S / I / D
    ref: str | None
    hyp: str | None


def align_sequences(ref: Sequence[T], hyp: Sequence[T]) -> tuple[EditCounts, list[tuple[str, T | None, T | None]]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
                continue
            sub, ins, delete = dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1
            best, op = sub, "S"
            if ins < best:
                best, op = ins, "I"
            if delete < best:
                best, op = delete, "D"
            dp[i][j] = best
            bt[i][j] = op

    i, j = n, m
    ops: list[tuple[str, T | None, T | None]] = []
    subs = ins_n = dels = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            ops.append(("M", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "S":
            ops.append(("S", ref[i - 1], hyp[j - 1]))
            subs += 1
            i -= 1
            j -= 1
        elif op == "I":
            ops.append(("I", None, hyp[j - 1]))
            ins_n += 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref[i - 1], None))
            dels += 1
            i -= 1
        else:
            if i > 0:
                ops.append(("D", ref[i - 1], None))
                dels += 1
                i -= 1
            else:
                ops.append(("I", None, hyp[j - 1]))
                ins_n += 1
                j -= 1
    ops.reverse()
    return EditCounts(subs, ins_n, dels), ops


def cer(ref_text: str, hyp_text: str, *, keep_punct: bool = False) -> dict:
    ref = normalize_for_cer(ref_text, keep_punct=keep_punct)
    hyp = normalize_for_cer(hyp_text, keep_punct=keep_punct)
    counts, ops = align_sequences(list(ref), list(hyp))
    denom = max(1, len(ref))
    return {
        "cer": counts.distance / denom,
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        **asdict(counts),
        "distance": counts.distance,
        "ops": [{"op": op, "ref": r, "hyp": h} for op, r, h in ops],
    }


def wer(ref_text: str, hyp_text: str) -> dict:
    ref_tokens = [t for t in ref_text.split() if t]
    hyp_tokens = [t for t in hyp_text.split() if t]
    counts, _ = align_sequences(ref_tokens, hyp_tokens)
    denom = max(1, len(ref_tokens))
    return {
        "wer": counts.distance / denom,
        "ref_words": len(ref_tokens),
        **asdict(counts),
        "distance": counts.distance,
    }
