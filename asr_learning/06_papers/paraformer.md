# Paraformer (FunASR, 2022)

贡献：非自回归，Predictor + 平行解码，CTC/时间戳天然对齐，中文生产常用。

生产哪一层：双模型里负责「轴」。PDF 原话接近「幻觉形态不同但时间轴准」。所以方案A 是 *Belle 文本 ⊕ Paraformer 词时间戳*，而不是指望 Whisper 自己的段时间。

样本轨怎么验证：看 IoU precision 已经不低（~90%），说明轴不是主溃点；MeanErr 用 `run_scheme_a.py`。若只换更大 Whisper 却不修 FA，报表上 recall 未必动。
