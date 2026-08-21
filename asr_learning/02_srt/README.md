# 02_srt · 字幕是管线的合同格式

生产对照：Gateway 吐给剪辑的 SRT，以及报表页用的 cue 统计。

为什么必须空行分块、时间必须是 `HH:MM:SS,mmm`：后续 IoU、CPS、方案A 全部依赖解析稳定。解析一松，所有指标都在撒谎。

```bash
python3 asr_learning/02_srt/srt_parser.py asr_learning/data/srt/origin.sample.srt | head
python3 asr_learning/02_srt/run_stats.py
```

看两件事：

- origin 223 段 vs asr 116 段
- CPS 均值：编辑在意「一行会不会闪」
