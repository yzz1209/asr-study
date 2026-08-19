#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE, ensure_results  # noqa: E402
from asr_learning.lib.metrics import cer  # noqa: E402
from asr_learning.lib.mock_decoder import DEFAULT_CONFIGS, mock_decode  # noqa: E402
from asr_learning.lib.srt import joined_text, load_srt  # noqa: E402
from asr_learning.lib.taxonomy import language_bottleneck_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=str(ORIGIN_SAMPLE))
    parser.add_argument("--asr", default=str(ASR_SAMPLE))
    args = parser.parse_args()
    origin = load_srt(args.origin)
    asr = load_srt(args.asr)
    ref = joined_text(origin)
    rows = []
    for cfg in DEFAULT_CONFIGS:
        dec = mock_decode(ref, cfg, seed=7)
        m = cer(ref, dec.text, keep_punct=False)
        rows.append(
            {
                "temperature": cfg.temperature,
                "beam_size": cfg.beam_size,
                "condition_on_previous_text": cfg.condition_on_previous_text,
                "cer": m["cer"],
                "hallucination_spans": dec.hallucination_spans,
                "latency_units": dec.latency_units,
                "avg_logprob": dec.avg_logprob,
                "no_speech_prob": dec.no_speech_prob,
            }
        )
    bottleneck = language_bottleneck_report(origin, asr)
    out = {
        "note": "mock decoder 只演示参数对指标的方向，不是真实 Whisper/Belle。真实消融要换 05_inference 里的模型入口。",
        "ablation": rows,
        "real_asr_vs_origin": {
            "cer_strip_punct": cer(ref, joined_text(asr), keep_punct=False)["cer"],
            "hotword_candidates_head": bottleneck["hotword_candidates"][:15],
        },
    }
    path = ensure_results() / "decoding_ablation.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
