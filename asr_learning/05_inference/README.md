# 05_inference · Belle / Paraformer 原理，不加载权重

生产对照：Whisper/Belle 解码参数、Paraformer CTC 时间戳、幻觉后处理。

本目录**故意不下载模型**。用 CTC 小实验 + mock decoder 把 PDF 任务 5/6/10 的指标表面跑出来，再用真实 `asr.sample.srt` 做幻觉候选。

```bash
python3 asr_learning/05_inference/run_ctc_demo.py
python3 asr_learning/05_inference/run_decoding_ablation.py
python3 asr_learning/05_inference/run_hallucination.py
```

| 旋钮 | 生产直觉 |
| --- | --- |
| `temperature=0` | 消随机，字幕必须可复现 |
| `beam_size=1` | greedy，更快更脏 |
| `beam_size=5~8` | 更稳，更慢 |
| `condition_on_previous_text=True` | 上一段幻觉会污染后面整集 |
| CTC greedy collapse | Paraformer 时间戳从这里来 |
| BoH + 无 overlap + n-gram + logprob | 任务 10 的多维过滤器 |

接到 GPU 之后，把 `mock_decode` 换成真正的 Belle/faster-whisper 调用，**报表字段不要改**。
