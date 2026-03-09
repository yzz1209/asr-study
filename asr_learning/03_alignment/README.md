# 03_alignment：时间轴对齐（第二阶段 B/C 任务）

这里的目标只有一件事：给定两份字幕（origin vs asr），用“时间轴”判断对齐好不好，并把问题定位到“漏段 / 乱切 / 固定偏移 / 漂移”。

## 文件职责（你只需要记住这三件事）

### 1) time_metrics.py：时间轴指标与匹配（B 任务的核心库）
- 解决的问题：两个时间段是否“对上了”
- 提供的能力：
  - interval_iou：两个区间的 IoU（重叠率）
  - greedy_match_by_iou：把 ref 段与 hyp 段按 IoU 做一对一匹配
  - alignment_pr：在 IoU 阈值下算 Precision/Recall

### 2) run_srt_alignment_report.py：把“字幕文件”接到指标库（B/C 任务入口脚本）
- 解决的问题：你不需要写代码就能跑报告/出图
- 做的事：
  - 读取两份 .srt
  - 调用 time_metrics.py 做匹配与统计
  - 输出 JSON 报告（precision/recall、偏移统计、最差段）
  - 可选输出时间轴图（timeline）

### 3) dtw.py：DTW 序列对齐（后续进阶，不是 B/C 的必需品）
- 解决的问题：当你不想按“段对段”匹配，而想对齐两条“序列”（例如每段的中点时间序列、特征序列）时，用 DTW 找一条整体一致的对齐路径
- 当前阶段（B/C）可以先不碰它

## 你现在该怎么用（只跑入口脚本即可）

在项目根目录执行：

```bash
python3 asr_learning/03_alignment/run_srt_alignment_report.py \
  --ref-srt asr_learning/data/srt/origin.sample.srt \
  --hyp-srt asr_learning/data/srt/asr.sample.srt \
  --min-iou 0.3 \
  --topk 5 \
  --plot
```

输出：
- 终端打印 JSON 报告
- `asr_learning/results/timeline_alignment.png` 时间轴图

## 如何读结果（最常用的 4 个信号）
- recall 低：origin 有很多段没被覆盖（常见是漏段/切段粒度差）
- precision 低：asr 有很多段找不到对应（常见是乱切/多余段）
- start_offset_ms_mean/median 非 0：整体固定偏移（整体早/晚）
- start_offset_ms 范围很大且随时间变：可能存在漂移（时基/分段累计误差）
