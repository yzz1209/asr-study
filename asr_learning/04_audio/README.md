# 04_audio · 16k PCM 和 VAD

生产对照：转封装锁 16kHz/16bit/mono，VAD 用 Silero；能量 VAD 是你必须能手写的基线。

```bash
python3 asr_learning/04_audio/make_demo_wav.py
python3 asr_learning/04_audio/vad_energy_wav.py asr_learning/data/raw_audio/demo_speech.wav
python3 asr_learning/04_audio/run_features.py
```

要能回答：

- 8kHz 会切掉哪些辅音能量，影视对白 CER 为什么掉
- 配乐轨上 RMS 阈值会把鼓点当说话，Silero 为什么更稳（学到的是语音/非语音表征，不是能量）
- Mel 三角滤波是 Whisper / Paraformer 的共同起点
