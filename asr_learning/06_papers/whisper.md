# Whisper (OpenAI, 2022)

贡献：多任务（转写/翻译/语言识别）Encoder-Decoder Transformer，zero-shot 跨数据。

生产哪一层：Belle 是中文数据上的 Whisper 系。文本质量往往好于 Paraformer，但自回归解码时间戳差、会幻觉、`condition_on_previous_text` 会级联。

样本轨怎么验证：不要一上来微调。先锁 `temperature=0`，对比 beam=1 vs 5 的 CER 和幻觉段数（`05_inference/run_decoding_ablation.py`）。PDF 要求能手画：波形 → Mel → encoder → decoder token → 可选 FA。
