# 07_training · 先热词，再谈 LoRA

生产对照：影视领域适配。通用 ASR 在古装玄幻轨上掉点，多半先死在词表。

```bash
python3 asr_learning/07_training/extract_hotwords.py
python3 asr_learning/07_training/run_spec_augment.py
```

ROI 口算：

1. 热词 / 上下文增强：人日级，专名召回通常先动
2. SpecAugment / 速度扰动：做微调才有意义
3. LoRA Whisper：要 GPU 和标注，只有热词吃不下的系统性错误才上
4. 换 Qwen-Audio-3.0：当语言瓶颈证明是模型语义能力而不是词表缺失
