# WhisperX (Bain et al., 2023)

贡献：先 ASR，再 wav2vec2 CTC 做音素级 forced alignment，切分更稳。

生产哪一层：方案A 的「重型备选」。WhisperX 要额外跑 CTC aligner；方案A 只用字符串匹配，便宜，MeanErr 门槛 800ms。清洁对白可以跳过 FA（自适应路由 `belle_only`）。

样本轨怎么验证：同一轨上对比方案A MeanErr vs（有 GPU 时）WhisperX。差值稳定小于 200ms 就别为 FA 加一台机器。
