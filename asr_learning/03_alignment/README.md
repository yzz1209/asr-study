# 03_alignment：时间轴对齐（生产：漏段 / 乱切 / 偏移 / 漂移 / 方案A）

给定 `origin.srt` vs `asr.srt`，先判断轴对齐，再谈文本 CER。

## 1) `time_metrics.py` / `lib/time_align.py`

- `interval_iou`：两段重叠率
- `greedy_match_by_iou`：一对一匹配
- `alignment_pr`：IoU 阈值下的 Precision / Recall
- `diagnose_alignment`：把数字翻译成漏段、乱切、固定偏移、漂移

## 2) `run_srt_alignment_report.py`

```bash
python3 asr_learning/03_alignment/run_srt_alignment_report.py \
  --ref-srt asr_learning/data/srt/origin.sample.srt \
  --hyp-srt asr_learning/data/srt/asr.sample.srt \
  --min-iou 0.3 --topk 5 --plot
```

样本上大约：precision 90%，recall 47%。ASR 段大多能对上，origin 一半对白没被覆盖。

## 3) `run_scheme_a.py`（A2 垂直路径）

PDF 任务 8：不用 WhisperX，把 Belle 类文本贴到 Paraformer 类字级时间戳上。

这里用 origin 文本当「较好文本」，用 asr cue 线性炸成字级时间戳，然后 `SequenceMatcher` 对齐，报 MeanErr（门槛 800ms）。

```bash
python3 asr_learning/03_alignment/run_scheme_a.py
```

## 4) `dtw.py`

段对段贪心不够时，用 DTW 对齐两条中点时间序列。WhisperX 的音素 FA 是另一条路，先把 DTW 矩阵手绘会。
