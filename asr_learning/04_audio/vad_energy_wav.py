from __future__ import annotations

from dataclasses import dataclass
import audioop
import json
import wave


@dataclass(frozen=True)
class VadSegment:
    start_ms: int
    end_ms: int


def _clamp_ms(x: int) -> int:
    return max(0, int(x))


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
        positions: list[int] = []

        pos = 0
        while pos < nframes:
            wf.setpos(pos)
            raw = wf.readframes(frame_len)
            if not raw:
                break
            rms = audioop.rms(raw, sampwidth) / max_amp
            speech_flags.append(rms >= rms_threshold)
            positions.append(pos)
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
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    end_frame = i - silence_run + 1
                    if end_frame - start_frame >= min_speech_frames:
                        start_ms = _clamp_ms(start_frame * hop_ms)
                        end_ms = _clamp_ms(end_frame * hop_ms + frame_ms)
                        segments.append(VadSegment(start_ms=start_ms, end_ms=end_ms))
                    in_speech = False
                    silence_run = 0

    if in_speech:
        end_frame = len(speech_flags) - 1
        if end_frame - start_frame + 1 >= min_speech_frames:
            start_ms = _clamp_ms(start_frame * hop_ms)
            end_ms = _clamp_ms(end_frame * hop_ms + frame_ms)
            segments.append(VadSegment(start_ms=start_ms, end_ms=end_ms))

    return segments


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--hop-ms", type=int, default=10)
    parser.add_argument("--rms-threshold", type=float, default=0.02)
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--min-silence-ms", type=int, default=200)
    args = parser.parse_args()

    segs = energy_vad_wav(
        args.wav_path,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        rms_threshold=args.rms_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
    )
    print(json.dumps([s.__dict__ for s in segs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
