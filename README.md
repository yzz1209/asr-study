# asr-study · 把真实字幕 ASR 拆开学

这份仓库对上 `1_ASR学习路径.pdf`，并用 `origin.sample.srt` / `asr.sample.srt` 当真实影视轨，而不是「你好世界」玩具句。

生产侧是 **Belle（文本）+ Paraformer（时间戳）+ 方案A 文本 FA + 幻觉后处理 + Gateway 报表**。这里用可跑的迷你实现把同一条链路走通，不加载大模型。

## 先跑

```bash
python3 -m pip install -r requirements.txt
python3 run_all_lessons.py
python3 -m pytest -q
python3 asr_learning/pipeline/run_asr_report.py
```

打开 `asr_learning/results/asr_report.html`。那就是学习版 ASR 报表页。

## 怎么学

1. 读 [`asr_learning/00_admin/生产管线对照.md`](asr_learning/00_admin/生产管线对照.md) —— 生产模块 ↔ 本仓库文件
2. 按 [`asr_learning/00_admin/阶段任务卡.md`](asr_learning/00_admin/阶段任务卡.md) 一周一张卡
3. 每个阶段用 [`asr_learning/00_admin/每周自查清单.txt`](asr_learning/00_admin/每周自查清单.txt) 过三问
4. 样本轨会告诉你：CER≈56% 主要是**删除/漏段**，时间轴 precision 仍然很高。先会读这两类数，再谈换 Qwen-Audio-3.0

## 目录

| 路径 | 生产对应 |
| --- | --- |
| `01_metrics` | CER 平台 + 语言瓶颈 |
| `02_srt` | 字幕规范 |
| `03_alignment` | IoU / 方案A（A2） / DTW |
| `04_audio` | 16k PCM + VAD |
| `05_inference` | Belle 解码消融 + CTC + 幻觉 |
| `06_papers` | 论文 → 管线哪一层 |
| `07_training` | 热词 / SpecAugment |
| `08_engineering` | 自适应路由 + Gateway |
| `pipeline` | 报表页一键生成 |
| `lib` | 共用实现 |

Python 需要 3.12（`audioop` 在 3.13 移除，VAD 依赖它）。
