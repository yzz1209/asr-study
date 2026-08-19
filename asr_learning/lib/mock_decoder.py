from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float
    beam_size: int
    condition_on_previous_text: bool = False


@dataclass(frozen=True)
class DecodeResult:
    text: str
    cer_proxy_noise: float
    hallucination_spans: int
    latency_units: float
    avg_logprob: float
    no_speech_prob: float
    config: DecodeConfig


# Typical Whisper-family Chinese hallucinations used in Belle logprob analysis (PDF 任务 6).
_HALLU = ("谢谢观看", "请不吝点赞", "字幕志愿者", "未完待续")


def mock_decode(ref_text: str, config: DecodeConfig, seed: int = 0) -> DecodeResult:
    """
    GPU-free stand-in for 任务5：temperature / beam_size 消融。
    Not a real decoder — it makes the *metric surface* of those knobs visible:
      temperature>0  → extra hallucination inserts
      beam_size=1    → more substitutions (greedy)
      beam_size>=5   → fewer substitutions, slightly slower
      condition_on_previous_text=True → cascading extra hallu
    """
    rng = random.Random(seed + int(config.temperature * 100) + config.beam_size)
    chars = list(ref_text)
    subs = 0
    # greedy (beam=1) substitutes more characters
    sub_p = 0.08 if config.beam_size <= 1 else 0.03
    out: list[str] = []
    for ch in chars:
        if ch.strip() and rng.random() < sub_p:
            out.append("的" if ch != "的" else "地")
            subs += 1
        else:
            out.append(ch)

    hallu_n = 0
    extra_p = config.temperature * 0.8
    if config.condition_on_previous_text:
        # Cascading: previous-text conditioning injects a span even at temperature 0.
        out.extend(list(_HALLU[0]))
        hallu_n += 1
    elif extra_p > 0 and rng.random() < extra_p:
        out.extend(list(_HALLU[rng.randrange(len(_HALLU))]))
        hallu_n += 1

    latency = 1.0 + 0.15 * max(1, config.beam_size) + 0.2 * (1 if config.temperature > 0 else 0)
    avg_lp = -0.2 - 0.4 * (subs / max(1, len(chars))) - 0.6 * hallu_n - 0.3 * config.temperature
    no_speech = min(0.99, 0.05 + 0.4 * hallu_n + 0.1 * config.temperature)
    return DecodeResult(
        text="".join(out),
        cer_proxy_noise=subs / max(1, len(chars)),
        hallucination_spans=hallu_n,
        latency_units=latency,
        avg_logprob=avg_lp,
        no_speech_prob=no_speech,
        config=config,
    )


DEFAULT_CONFIGS = [
    DecodeConfig(temperature=0.0, beam_size=5, condition_on_previous_text=False),
    DecodeConfig(temperature=0.2, beam_size=5, condition_on_previous_text=False),
    DecodeConfig(temperature=0.0, beam_size=1, condition_on_previous_text=False),
    DecodeConfig(temperature=0.0, beam_size=5, condition_on_previous_text=True),
]
