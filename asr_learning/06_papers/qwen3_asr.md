# Qwen-Audio-3.0 / Qwen3-ASR

这是你们评估过的候选替换，不是当前学习仓的运行时依赖。

Qwen3-ASR（0.6B/1.7B）带语种识别、长音频、热词/上下文增强；云端还有 `qwen-audio-3.0-asr-flash` 与 filetrans。它和 Belle 的差别主要在：

- 上下文/热词是一等公民，对影视专名（语言瓶颈）可能比「再调 Whisper 解码」更直接
- 时间戳和 FA 生态不如 Paraformer + WhisperX 成熟，**换模型不等于可以拆掉方案A**
- 成本与 halluc 形态会变，必须用同一套报表字段做 A/B：CER 去标点、IoU P/R、MeanErr、幻觉条数、专名召回

样本轨怎么验证（有 API 再做）：

1. 同一音轨 Belle vs Qwen，只换识别，不换 VAD/FA/后处理
2. 热词表用 `07_training/extract_hotwords.py` 从 origin 删除片段生成，两边都灌
3. 报表页字段不变，禁止换一套指标宣布胜利
