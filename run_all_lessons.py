#!/usr/bin/env python3
"""Run the miniature production loop on the sample SRTs (and demo wav)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.check_call(args, cwd=str(REPO))


def main() -> None:
    run([PY, "asr_learning/01_metrics/cer_manual.py", "--ref-srt", "asr_learning/data/srt/origin.sample.srt", "--hyp-srt", "asr_learning/data/srt/asr.sample.srt"])
    run([PY, "asr_learning/01_metrics/run_error_analysis.py"])
    run([PY, "asr_learning/02_srt/run_stats.py"])
    run(
        [
            PY,
            "asr_learning/03_alignment/run_srt_alignment_report.py",
            "--ref-srt",
            "asr_learning/data/srt/origin.sample.srt",
            "--hyp-srt",
            "asr_learning/data/srt/asr.sample.srt",
            "--min-iou",
            "0.3",
            "--topk",
            "3",
        ]
    )
    run([PY, "asr_learning/03_alignment/run_scheme_a.py"])
    run([PY, "asr_learning/04_audio/make_demo_wav.py"])
    run([PY, "asr_learning/04_audio/run_features.py"])
    run([PY, "asr_learning/05_inference/run_ctc_demo.py"])
    run([PY, "asr_learning/05_inference/run_decoding_ablation.py"])
    run([PY, "asr_learning/05_inference/run_hallucination.py"])
    run([PY, "asr_learning/07_training/extract_hotwords.py"])
    run([PY, "asr_learning/07_training/run_spec_augment.py"])
    run([PY, "asr_learning/08_engineering/run_adaptive_pipeline.py"])
    run([PY, "asr_learning/pipeline/run_asr_report.py"])
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
