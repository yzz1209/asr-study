from __future__ import annotations

import math


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, sr: int, n_mels: int = 40, fmin: float = 0.0, fmax: float | None = None) -> list[list[float]]:
    """Triangular Mel filters. Production ASR (Whisper/Paraformer) starts from a Mel spectrogram."""
    fmax = fmax if fmax is not None else sr / 2.0
    n_bins = n_fft // 2 + 1
    mels = [hz_to_mel(fmin) + i * (hz_to_mel(fmax) - hz_to_mel(fmin)) / (n_mels + 1) for i in range(n_mels + 2)]
    hz = [mel_to_hz(m) for m in mels]
    bins = [int(f / (sr / n_fft)) for f in hz]
    fb = [[0.0] * n_bins for _ in range(n_mels)]
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if right <= left:
            continue
        for k in range(left, center):
            if 0 <= k < n_bins and center != left:
                fb[m - 1][k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < n_bins and right != center:
                fb[m - 1][k] = (right - k) / (right - center)
    return fb


def spec_augment_mask(n_frames: int, n_mels: int, time_mask: int = 8, freq_mask: int = 4, seed: int = 0) -> list[list[int]]:
    """Return a 0/1 mask (1=keep). SpecAugment is the cheap domain-robustness trick before LoRA."""
    rng_state = seed
    def rnd(n: int) -> int:
        nonlocal rng_state
        rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
        return rng_state % max(1, n)

    mask = [[1 for _ in range(n_mels)] for _ in range(n_frames)]
    t0 = rnd(max(1, n_frames - time_mask))
    for t in range(t0, min(n_frames, t0 + time_mask)):
        for f in range(n_mels):
            mask[t][f] = 0
    f0 = rnd(max(1, n_mels - freq_mask))
    for t in range(n_frames):
        for f in range(f0, min(n_mels, f0 + freq_mask)):
            mask[t][f] = 0
    return mask
