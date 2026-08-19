# Calm-Whisper (arXiv 2505.12969)

贡献：幻觉和部分 attention head 相关，微调/抑制这些 head 能降幻觉。

生产哪一层：后处理之上的模型内干预。PDF 任务 9 不要求完整微调：抽出 attention，对比幻觉段 vs 正常段，看哪些 head 异常。

样本轨怎么验证：先用 `run_hallucination.py` 列出无 overlap / BoH 段，作为「幻觉段索引」；有权重后再挂 hook。没有 GPU 时，任务 10 的 BoH+重复+logprob 仍然是上线必须有的安全网。
