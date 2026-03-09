from __future__ import annotations

import argparse
import json
import os
import sys

from cer_manual import calculate_cer_manual, extract_text_from_srt_path


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_auto(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    if ext == ".srt":
        return extract_text_from_srt_path(path)
    return _read_text_file(path)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-path", required=True)
    parser.add_argument("--hyp-path", required=True)
    parser.add_argument("--keep-spaces", action="store_true")
    args = parser.parse_args()

    ref_text = load_text_auto(args.ref_path)
    hyp_text = load_text_auto(args.hyp_path)
    cer, counts = calculate_cer_manual(ref_text, hyp_text, remove_spaces=not args.keep_spaces)

    out = {
        "cer": cer,
        "substitutions": counts.substitutions,
        "insertions": counts.insertions,
        "deletions": counts.deletions,
        "distance": counts.distance,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    _main()
