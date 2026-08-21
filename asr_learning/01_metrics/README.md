# 01_metrics · 文本指标和语言瓶颈

生产对照：质量平台上的 CER，以及「ASR language bottleneck」讨论。

这份样本 `cer_manual.py` 算出来大约：

- CER ≈ 0.56
- 替换 102 / 插入 76 / **删除 572**

删除远大于替换，说明主因是漏了 origin 对白，不是「灵相」写成「灵想」那种单字错。单字错仍然重要，它走热词，不走「再买一个模型」。

```bash
python3 asr_learning/01_metrics/cer_manual.py \
  --ref-srt asr_learning/data/srt/origin.sample.srt \
  --hyp-srt asr_learning/data/srt/asr.sample.srt

python3 asr_learning/01_metrics/run_error_analysis.py
```

`run_error_analysis.py` 会：

1. 对比「去标点 CER」和「带标点 CER」（标点模型不能拿带标点 CER 打无标点模型）
2. 列出高频替换
3. 把连续删除片段当成热词候选（地名、境界名、人名）
