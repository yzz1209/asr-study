from __future__ import annotations

from dataclasses import dataclass
import audioop
import math
import struct
import wave
from pathlib import Path


@dataclass(frozen=True)
class VadSegment:
    start_ms: int
    end_ms: int


def write_sine_wav(path: str | Path, events: list[tuple[float, float, float]], sr: int = 16000) -> None:
    """events: (start_sec, end_sec, freq_hz). Gaps are silence. 16-bit mono PCM."""
    if not events:
        raise ValueError("events required")
    duration = max(end for _, end, _ in events)
    n = int(sr * duration) + sr // 10
    samples = [0.0] * n
    for start, end, freq in events:
        a, b = int(start * sr), int(end * sr)
        for i in range(a, min(b, n)):
            t = i / sr
            samples[i] = 0.35 * math.sin(2 * math.pi * freq * t)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767)) for x in samples)
        wf.writeframes(frames)


def energy_vad_wav(
    wav_path: str,
    frame_ms: int = 30,
    hop_ms: int = 10,
    rms_threshold: float = 0.02,
    min_speech_ms: int = 200,
    min_silence_ms: int = 200,
) -> list[VadSegment]:
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        if channels != 1:
            raise ValueError(f"Only mono wav supported, got channels={channels}")
        if sampwidth != 2:
            raise ValueError(f"Only 16-bit wav supported, got sampwidth={sampwidth}")
        frame_len = max(1, int(sr * frame_ms / 1000))
        hop_len = max(1, int(sr * hop_ms / 1000))
        max_amp = float(2**15)
        speech_flags: list[bool] = []
        pos = 0
        while pos < nframes:
            wf.setpos(pos)
            raw = wf.readframes(frame_len)
            if not raw:
                break
            rms = audioop.rms(raw, sampwidth) / max_amp
            speech_flags.append(rms >= rms_threshold)
            pos += hop_len

    if not speech_flags:
        return []

    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    min_silence_frames = max(1, int(min_silence_ms / hop_ms))
    segments: list[VadSegment] = []
    in_speech = False
    start_frame = 0
    silence_run = 0
    for i, is_speech in enumerate(speech_flags):
        if is_speech:
            silence_run = 0
            if not in_speech:
                in_speech = True
                start_frame = i
        elif in_speech:
            silence_run += 1
            if silence_run >= min_silence_frames:
                end_frame = i - silence_run + 1
                if end_frame - start_frame >= min_speech_frames:
                    start_ms = max(0, start_frame * hop_ms)
                    end_ms = max(0, end_frame * hop_ms + frame_ms)
                    segments.append(VadSegment(start_ms=start_ms, end_ms=end_ms))
                in_speech = False
                silence_run = 0
    if in_speech:
        end_frame = len(speech_flags) - 1
        if end_frame - start_frame + 1 >= min_speech_frames:
            start_ms = max(0, start_frame * hop_ms)
            end_ms = max(0, end_frame * hop_ms + frame_ms)
            segments.append(VadSegment(start_ms=start_ms, end_ms=end_ms))
    return segments
