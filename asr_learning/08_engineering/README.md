# 08_engineering · 自适应路由、性能、Gateway

生产对照：智能降级（PDF 任务 14）、profiling、Gateway、报表页。

```bash
python3 asr_learning/08_engineering/run_adaptive_pipeline.py
python3 asr_learning/08_engineering/run_profile.py
python3 asr_learning/08_engineering/mini_gateway.py --port 8765
# 浏览器打开 http://127.0.0.1:8765/report
# JSON   GET  http://127.0.0.1:8765/api/v1/asr/report
```

路由：

| 猜测 | 动作 |
| --- | --- |
| 对白密、节奏稳 | `belle_only` 跳过 FA |
| 一般噪声 / 切段不稳 | `paraformer_plus_fa`（A2） |
| 稀疏对白 + 长静音（配乐） | `belle_aggressive_filter` |

误路由代价：清洁轨走强过滤会误杀对白；配乐轨跳过过滤会出「谢谢观看」。
