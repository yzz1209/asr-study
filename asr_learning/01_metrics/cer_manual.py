from __future__ import annotations

from dataclasses import dataclass
import re


STAGE_1_GOAL = {
    "name": "阶段一（Week 1-2）：理解数据流和评估指标",
    "deliverables": [
        "能从零实现 CER（手写编辑距离，不依赖第三方库）",
        "能把两份字幕（origin vs 生成）做定量对比：CER + 编辑操作统计",
    ],
    "done_when": [
        "能解释：CER 如何由插入/删除/替换组成",
        "能跑通：对 01-origin.srt 与 01.srt 输出一份 JSON 结果",
    ],
}


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions


def normalize_text(text: str, remove_spaces: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if remove_spaces:
        text = "".join(ch for ch in text if not ch.isspace())
    return text


def levenshtein_counts(ref: str, hyp: str) -> EditCounts:
    r = list(ref)
    h = list(hyp)
    n = len(r)
    m = len(h)
    # CER 的编辑距离（字符级）
    # dp[i][j] 表示把 ref[:i] 变成 hyp[:j] 的最小编辑代价
    # - M（match）：ref[i-1] == hyp[j-1]，走 dp[i-1][j-1]，代价 +0
    # - S（substitution）：来自 dp[i-1][j-1] + 1，把 ref[i-1] 替换为 hyp[j-1]
    # - I（insertion）：来自 dp[i][j-1] + 1，给 ref 插入 hyp[j-1]（等价于 hyp 多了一个 token）
    # - D（deletion）：来自 dp[i-1][j] + 1，删除 ref[i-1]（等价于 hyp 少了一个 token）
    #
    # WER 的计算过程完全一致，只是把“字符序列”换成“词序列”：
    # 1) 把 ref/hyp 分词得到 ref_tokens/hyp_tokens
    # 2) 在 token 序列上跑同样的 DP，统计 S/I/D
    # 3) WER = (S + I + D) / max(1, len(ref_tokens))
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
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
                continue
            # 替换
            sub_cost = dp[i - 1][j - 1] + 1
            # 插入
            ins_cost = dp[i][j - 1] + 1
            # 删除
            del_cost = dp[i - 1][j] + 1

            best = sub_cost
            op = "S"
            if ins_cost < best:
                best = ins_cost
                op = "I"
            if del_cost < best:
                best = del_cost
                op = "D"

            dp[i][j] = best
            bt[i][j] = op

    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            subs += 1
            i -= 1
            j -= 1
        elif op == "I":
            ins += 1
            j -= 1
        elif op == "D":
            dels += 1
            i -= 1
        else:
            if i > 0:
                dels += 1
                i -= 1
            elif j > 0:
                ins += 1
                j -= 1

    return EditCounts(substitutions=subs, insertions=ins, deletions=dels)


def calculate_cer_manual(ref_text: str, hyp_text: str, remove_spaces: bool = True) -> tuple[float, EditCounts]:
    ref = normalize_text(ref_text, remove_spaces=remove_spaces)
    hyp = normalize_text(hyp_text, remove_spaces=remove_spaces)
    counts = levenshtein_counts(ref, hyp)
    denom = max(1, len(ref))
    return counts.distance / denom, counts


def levenshtein_counts_tokens(ref_tokens: list[str], hyp_tokens: list[str]) -> EditCounts:
    n = len(ref_tokens)
    m = len(hyp_tokens)

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
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
                continue

            sub_cost = dp[i - 1][j - 1] + 1
            ins_cost = dp[i][j - 1] + 1
            del_cost = dp[i - 1][j] + 1

            best = sub_cost
            op = "S"
            if ins_cost < best:
                best = ins_cost
                op = "I"
            if del_cost < best:
                best = del_cost
                op = "D"

            dp[i][j] = best
            bt[i][j] = op

    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            subs += 1
            i -= 1
            j -= 1
        elif op == "I":
            ins += 1
            j -= 1
        elif op == "D":
            dels += 1
            i -= 1
        else:
            if i > 0:
                dels += 1
                i -= 1
            elif j > 0:
                ins += 1
                j -= 1

    return EditCounts(substitutions=subs, insertions=ins, deletions=dels)


def tokenize_words_whitespace(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split(" ")


def calculate_wer_manual(ref_text: str, hyp_text: str) -> tuple[float, EditCounts, int]:
    ref_tokens = tokenize_words_whitespace(ref_text)
    hyp_tokens = tokenize_words_whitespace(hyp_text)
    counts = levenshtein_counts_tokens(ref_tokens, hyp_tokens)
    denom = max(1, len(ref_tokens))
    return counts.distance / denom, counts, len(ref_tokens)


_SRT_TIMECODE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}.*$"
)


def extract_text_from_srt_content(content: str) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if _SRT_TIMECODE_RE.match(stripped):
            continue
        out_lines.append(stripped)
    return "\n".join(out_lines)


def extract_text_from_srt_path(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return extract_text_from_srt_content(f.read())


def _main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-goal", action="store_true")
    parser.add_argument("--ref")
    parser.add_argument("--hyp")
    parser.add_argument("--ref-srt")
    parser.add_argument("--hyp-srt")
    parser.add_argument("--wer", action="store_true")
    parser.add_argument("--keep-spaces", action="store_true")
    args = parser.parse_args()

    if args.print_goal:
        sys.stdout.write(json.dumps(STAGE_1_GOAL, ensure_ascii=False, indent=2) + "\n")
        return

    if (args.ref is None) != (args.hyp is None):
        raise SystemExit("--ref 与 --hyp 必须同时提供")
    if (args.ref_srt is None) != (args.hyp_srt is None):
        raise SystemExit("--ref-srt 与 --hyp-srt 必须同时提供")
    if (args.ref is not None) and (args.ref_srt is not None):
        raise SystemExit("请选择一种输入方式：--ref/--hyp 或 --ref-srt/--hyp-srt")
    if (args.ref is None) and (args.ref_srt is None):
        raise SystemExit("请提供 --ref/--hyp 或 --ref-srt/--hyp-srt")

    if args.ref_srt is not None:
        ref_text = extract_text_from_srt_path(args.ref_srt)
        hyp_text = extract_text_from_srt_path(args.hyp_srt)  # type: ignore[arg-type]
    else:
        ref_text = args.ref  # type: ignore[assignment]
        hyp_text = args.hyp  # type: ignore[assignment]

    if args.wer:
        wer, counts, ref_words = calculate_wer_manual(ref_text, hyp_text)
        out = {
            "wer": wer,
            "substitutions": counts.substitutions,
            "insertions": counts.insertions,
            "deletions": counts.deletions,
            "distance": counts.distance,
            "ref_words": ref_words,
        }
    else:
        cer, counts = calculate_cer_manual(ref_text, hyp_text, remove_spaces=not args.keep_spaces)
        out = {
            "cer": cer,
            "substitutions": counts.substitutions,
            "insertions": counts.insertions,
            "deletions": counts.deletions,
            "distance": counts.distance,
            "ref_len": len(normalize_text(ref_text, remove_spaces=not args.keep_spaces)),
        }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    _main()
