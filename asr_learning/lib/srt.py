from __future__ import annotations

from dataclasses import dataclass
import re

_TIMECODE_RE = re.compile(
    r"^(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}),(?P<ms1>\d{3})\s+-->\s+"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})\s*$"
)


@dataclass(frozen=True)
class SrtSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def cps(self) -> float:
        chars = len("".join(self.text.split()))
        sec = max(0.001, self.duration_ms / 1000.0)
        return chars / sec


def timecode_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def ms_to_timecode(ms: int) -> str:
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def parse_timecode_line(line: str) -> tuple[int, int]:
    m = _TIMECODE_RE.match(line.strip())
    if not m:
        raise ValueError(f"Invalid timecode line: {line!r}")
    start_ms = timecode_to_ms(int(m.group("h1")), int(m.group("m1")), int(m.group("s1")), int(m.group("ms1")))
    end_ms = timecode_to_ms(int(m.group("h2")), int(m.group("m2")), int(m.group("s2")), int(m.group("ms2")))
    return start_ms, end_ms


def parse_srt(content: str) -> list[SrtSegment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []
    blocks = re.split(r"\n{2,}", content)
    segments: list[SrtSegment] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
            time_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            idx = len(segments) + 1
            time_line = lines[0]
            text_lines = lines[1:]
        start_ms, end_ms = parse_timecode_line(time_line)
        text = "\n".join(text_lines).strip()
        segments.append(SrtSegment(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))
    return segments


def load_srt(path: str) -> list[SrtSegment]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_srt(f.read())


def dump_srt(segments: list[SrtSegment]) -> str:
    parts: list[str] = []
    for i, seg in enumerate(segments, start=1):
        parts.append(
            f"{i}\n{ms_to_timecode(seg.start_ms)} --> {ms_to_timecode(seg.end_ms)}\n{seg.text}\n"
        )
    return "\n".join(parts).rstrip() + "\n"


def joined_text(segments: list[SrtSegment]) -> str:
    return "\n".join(s.text for s in segments)


def subtitle_stats(segments: list[SrtSegment]) -> dict:
    if not segments:
        return {"segments": 0}
    durs = [s.duration_ms for s in segments]
    gaps = [segments[i].start_ms - segments[i - 1].end_ms for i in range(1, len(segments))]
    cps = [s.cps for s in segments]
    return {
        "segments": len(segments),
        "span_ms": segments[-1].end_ms - segments[0].start_ms,
        "duration_ms_mean": sum(durs) / len(durs),
        "duration_ms_max": max(durs),
        "gap_ms_mean": (sum(gaps) / len(gaps)) if gaps else 0,
        "cps_mean": sum(cps) / len(cps),
        "cps_max": max(cps),
        "chars": sum(len("".join(s.text.split())) for s in segments),
    }
