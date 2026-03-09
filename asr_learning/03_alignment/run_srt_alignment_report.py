from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, median

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "02_srt"))
if SRT_DIR not in sys.path:
    sys.path.insert(0, SRT_DIR)

from srt_parser import load_srt

from time_metrics import TimeSegment, alignment_pr, greedy_match_by_iou


def _to_time_segments(srt_segments) -> list[TimeSegment]:
    out: list[TimeSegment] = []
    for s in srt_segments:
        out.append(TimeSegment(index=int(s.index), start_ms=int(s.start_ms), end_ms=int(s.end_ms)))
    return out


def _summarize_matches(matches) -> dict:
    if not matches:
        return {"matched": 0}

    ious = [float(m.iou) for m in matches]
    offsets_start = [int(m.hyp_start_ms - m.ref_start_ms) for m in matches]
    offsets_end = [int(m.hyp_end_ms - m.ref_end_ms) for m in matches]
    durations_ref = [int(m.ref_end_ms - m.ref_start_ms) for m in matches]
    durations_hyp = [int(m.hyp_end_ms - m.hyp_start_ms) for m in matches]
    duration_delta = [h - r for h, r in zip(durations_hyp, durations_ref)]

    return {
        "matched": len(matches),
        "iou_mean": mean(ious),
        "iou_median": median(ious),
        "iou_min": min(ious),
        "start_offset_ms_mean": mean(offsets_start),
        "start_offset_ms_median": median(offsets_start),
        "start_offset_ms_min": min(offsets_start),
        "start_offset_ms_max": max(offsets_start),
        "end_offset_ms_mean": mean(offsets_end),
        "end_offset_ms_median": median(offsets_end),
        "duration_delta_ms_mean": mean(duration_delta),
        "duration_delta_ms_median": median(duration_delta),
    }


def _worst_matches(matches, topk: int) -> dict:
    worst_by_iou = sorted(matches, key=lambda m: (m.iou, abs(m.hyp_start_ms - m.ref_start_ms)))[:topk]
    worst_by_start_offset = sorted(matches, key=lambda m: abs(m.hyp_start_ms - m.ref_start_ms), reverse=True)[:topk]
    return {
        "worst_by_iou": [m.__dict__ for m in worst_by_iou],
        "worst_by_start_offset": [
            {
                **m.__dict__,
                "start_offset_ms": int(m.hyp_start_ms - m.ref_start_ms),
                "end_offset_ms": int(m.hyp_end_ms - m.ref_end_ms),
            }
            for m in worst_by_start_offset
        ],
    }


def _plot_timeline(ref: list[TimeSegment], hyp: list[TimeSegment], matches, out_path: str) -> None:
    mpl_dir = os.path.join("/tmp", "matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    import matplotlib.pyplot as plt

    def to_bars(segs: list[TimeSegment]):
        bars = []
        for s in segs:
            bars.append((s.start_ms / 1000.0, max(0.0, (s.end_ms - s.start_ms) / 1000.0)))
        return bars

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4))
    ref_bars = to_bars(ref)
    hyp_bars = to_bars(hyp)

    ax.broken_barh(ref_bars, (20, 6), facecolors="#1f77b4", alpha=0.7)
    ax.broken_barh(hyp_bars, (10, 6), facecolors="#ff7f0e", alpha=0.7)

    ax.set_yticks([13, 23])
    ax.set_yticklabels(["hyp", "ref"])
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, 35)
    ax.grid(True, axis="x", alpha=0.2)

    if matches:
        ref_map = {s.index: s for s in ref}
        hyp_map = {s.index: s for s in hyp}
        for m in matches:
            r = ref_map.get(m.ref_index)
            h = hyp_map.get(m.hyp_index)
            if r is None or h is None:
                continue
            xr = (r.start_ms + r.end_ms) / 2000.0
            xh = (h.start_ms + h.end_ms) / 2000.0
            ax.plot([xr, xh], [23, 13], color="black", alpha=0.05, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-srt", required=True)
    parser.add_argument("--hyp-srt", required=True)
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--dump-matches", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path")
    args = parser.parse_args()

    ref = _to_time_segments(load_srt(args.ref_srt))
    hyp = _to_time_segments(load_srt(args.hyp_srt))
    precision, recall = alignment_pr(ref, hyp, min_iou=args.min_iou)
    matches = greedy_match_by_iou(ref, hyp, min_iou=args.min_iou)
    used_ref = {m.ref_index for m in matches}
    used_hyp = {m.hyp_index for m in matches}

    out = {
        "precision": precision,
        "recall": recall,
        "ref_segments": len(ref),
        "hyp_segments": len(hyp),
        "unmatched_ref": len(ref) - len(used_ref),
        "unmatched_hyp": len(hyp) - len(used_hyp),
        "min_iou": args.min_iou,
        "match_summary": _summarize_matches(matches),
        "match_worst": _worst_matches(matches, topk=args.topk),
    }

    if args.dump_matches:
        out["matches"] = [m.__dict__ for m in matches]

    if args.plot:
        if args.plot_path:
            plot_path = args.plot_path
        else:
            plot_path = os.path.join(CURRENT_DIR, "..", "results", "timeline_alignment.png")
            plot_path = os.path.normpath(plot_path)
        _plot_timeline(ref, hyp, matches, plot_path)
        out["plot_path"] = plot_path

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
