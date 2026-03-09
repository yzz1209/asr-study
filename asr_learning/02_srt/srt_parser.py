from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SrtSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIMECODE_RE = re.compile(
    r"^(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}),(?P<ms1>\d{3})\s+-->\s+"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})\s*$"
)


def timecode_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_timecode_line(line: str) -> tuple[int, int]:
    m = _TIMECODE_RE.match(line.strip())
    if not m:
        raise ValueError(f"Invalid timecode line: {line!r}")
    start_ms = timecode_to_ms(
        int(m.group("h1")),
        int(m.group("m1")),
        int(m.group("s1")),
        int(m.group("ms1")),
    )
    end_ms = timecode_to_ms(
        int(m.group("h2")),
        int(m.group("m2")),
        int(m.group("s2")),
        int(m.group("ms2")),
    )
    return start_ms, end_ms


def parse_srt(content: str) -> list[SrtSegment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []

    blocks = re.split(r"\n{2,}", content)
    segments: list[SrtSegment] = []
    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = len(segments) + 1
            time_line = lines[0]
            text_lines = lines[1:]
        else:
            time_line = lines[1]
            text_lines = lines[2:]

        start_ms, end_ms = parse_timecode_line(time_line)
        text = "\n".join(text_lines).strip()
        segments.append(SrtSegment(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))
    return segments


def load_srt(path: str) -> list[SrtSegment]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_srt(f.read())


def _main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("srt_path")
    args = parser.parse_args()

    segs = load_srt(args.srt_path)
    sys.stdout.write(json.dumps([s.__dict__ for s in segs], ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    _main()
