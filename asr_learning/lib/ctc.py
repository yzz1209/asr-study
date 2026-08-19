from __future__ import annotations

BLANK = "<blank>"


def ctc_greedy_decode(frame_tokens: list[str], blank: str = BLANK) -> str:
    """
    Collapse CTC: remove blanks and collapse repeats.
    Paraformer is NAR/Predictor-CTC family — timestamps come from this alignment, which is why
    the PDF says「幻觉多但时间轴准」relative to autoregressive Whisper/Belle.
    """
    out: list[str] = []
    prev = None
    for tok in frame_tokens:
        if tok == blank:
            prev = blank
            continue
        if tok != prev:
            out.append(tok)
        prev = tok
    return "".join(out)


def ctc_forced_alignment(text: str, frame_tokens: list[str], blank: str = BLANK) -> list[tuple[str, int, int]]:
    """
    Tiny forced alignment: find a monotone path that emits `text` from CTC frames.
    Returns (char, start_frame, end_frame). Not a full HMM FA — enough to see the idea of WhisperX.
    """
    chars = list(text)
    if not chars:
        return []
    i = 0
    spans: list[tuple[str, int, int]] = []
    start = None
    for t, tok in enumerate(frame_tokens):
        if i >= len(chars):
            break
        if tok == blank:
            if start is not None:
                spans.append((chars[i], start, t - 1))
                i += 1
                start = None
            continue
        if tok == chars[i]:
            if start is None:
                start = t
        elif start is not None:
            spans.append((chars[i], start, t - 1))
            i += 1
            start = t if i < len(chars) and tok == chars[i] else None
    if start is not None and i < len(chars):
        spans.append((chars[i], start, len(frame_tokens) - 1))
    return spans
