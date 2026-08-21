from __future__ import annotations

import html
import json
from pathlib import Path

from .adaptive import guess_route
from .hallucination import detect_hallucinations, hits_to_dicts
from .metrics import cer
from .paths import ASR_SAMPLE, ORIGIN_SAMPLE, ensure_results
from .srt import joined_text, load_srt, subtitle_stats
from .taxonomy import language_bottleneck_report
from .text_align import scheme_a_report
from .time_align import diagnose_alignment


def build_report(origin_path: str | Path | None = None, asr_path: str | Path | None = None) -> dict:
    origin = load_srt(str(origin_path or ORIGIN_SAMPLE))
    asr = load_srt(str(asr_path or ASR_SAMPLE))
    ref_text, hyp_text = joined_text(origin), joined_text(asr)
    cer_raw = cer(ref_text, hyp_text, keep_punct=True)
    cer_clean = cer(ref_text, hyp_text, keep_punct=False)
    align = diagnose_alignment(origin, asr, min_iou=0.3)
    bottleneck = language_bottleneck_report(origin, asr)
    scheme_a = scheme_a_report(origin, asr)
    hits = detect_hallucinations(asr, origin)
    hallu = [h for h in hits if h.reasons != ["time_mismatch"]]
    timing = [h for h in hits if h.reasons == ["time_mismatch"]]
    route = guess_route(origin, asr)
    report = {
        "title": "ASR 字幕质量报告（学习版 / 对照真实管线）",
        "inputs": {
            "origin_srt": str(origin_path or ORIGIN_SAMPLE),
            "asr_srt": str(asr_path or ASR_SAMPLE),
        },
        "pipeline": [
            "音频 PCM 16k/16bit",
            "VAD（生产 Silero / 学习能量阈值）",
            "双模型：Belle(Whisper Attention 文本) + Paraformer(CTC/NAR 时间戳)",
            "A2 垂直路径：方案A 文本级 FA（Belle 文本贴 Paraformer 时间戳）",
            "幻觉过滤：BoH / 无重叠 / n-gram / logprob",
            "Gateway 出 SRT + 报表页",
        ],
        "text_metrics": {
            "cer_keep_punct": cer_raw["cer"],
            "cer_strip_punct": cer_clean["cer"],
            "substitutions": cer_clean["substitutions"],
            "insertions": cer_clean["insertions"],
            "deletions": cer_clean["deletions"],
            "ref_len": cer_clean["ref_len"],
            "hyp_len": cer_clean["hyp_len"],
        },
        "origin_stats": subtitle_stats(origin),
        "asr_stats": subtitle_stats(asr),
        "alignment": align,
        "scheme_a": scheme_a,
        "language_bottleneck": bottleneck,
        "hallucinations": hits_to_dicts(hallu)[:30],
        "hallucination_count": len(hallu),
        "timing_mismatches": hits_to_dicts(timing)[:20],
        "timing_mismatch_count": len(timing),
        "adaptive_route": {
            "route": route.route,
            "reason": route.reason,
            "speech_density": route.speech_density,
            "cue_cps_mean": route.cue_cps_mean,
            "long_gap_ratio": route.long_gap_ratio,
        },
        "readout": _readout(cer_clean["cer"], align, bottleneck, scheme_a, len(hallu), len(timing)),
    }
    return report


def _readout(cer_v: float, align: dict, bottleneck: dict, scheme_a: dict, n_hallu: int, n_timing: int) -> list[str]:
    lines = []
    lines.append(
        f"CER(去标点)={cer_v:.1%} 而时间轴 recall={align['recall']:.1%} / precision={align['precision']:.1%}。"
        " CER 高 + recall 低 = 漏了大段对白，不是单纯同音错字。"
    )
    lines.extend(align.get("failure_modes") or [])
    if bottleneck.get("hotword_candidates"):
        sample = "、".join(bottleneck["hotword_candidates"][:8])
        lines.append(f"语言层瓶颈候选热词：{sample}")
    mean_err = scheme_a.get("mean_err_ms")
    if mean_err is not None:
        flag = "达标" if scheme_a.get("pass_mean_err_800ms") else "未达标"
        lines.append(f"方案A MeanErr={mean_err:.0f}ms（PDF 门槛 800ms，{flag}）")
    lines.append(
        f"字面对得上但轴对不上 {n_timing} 条（切段/IoU，不是幻觉）；"
        f"真幻觉候选 {n_hallu} 条（无 origin 文本或 BoH/重复）。"
    )
    return lines


def write_json(report: dict, path: Path | None = None) -> Path:
    out = path or (ensure_results() / "asr_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_html(report: dict, path: Path | None = None) -> Path:
    out = path or (ensure_results() / "asr_report.html")
    tm = report["text_metrics"]
    al = report["alignment"]
    sa = report["scheme_a"]
    bn = report["language_bottleneck"]
    route = report["adaptive_route"]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def table(rows: list[tuple[str, str]]) -> str:
        body = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        return f'<table class="kv">{body}</table>'

    subst = "".join(
        f"<li><code>{html.escape(x['ref'])}</code> → <code>{html.escape(x['hyp'])}</code> ×{x['n']}</li>"
        for x in bn.get("top_substitutions", [])[:10]
    )
    hotwords = "".join(
        f"<li><code>{html.escape(w)}</code></li>" for w in bn.get("hotword_candidates", [])[:20]
    )
    hallu = "".join(
        f"<li>#{h['index']} {html.escape(', '.join(h['reasons']))}: {html.escape(h['text'][:80])}</li>"
        for h in report.get("hallucinations", [])[:12]
    )
    timing = "".join(
        f"<li>#{h['index']}: {html.escape(h['text'][:80])}</li>"
        for h in report.get("timing_mismatches", [])[:12]
    )
    readout = "".join(f"<li>{html.escape(x)}</li>" for x in report.get("readout", []))
    pipeline = " → ".join(html.escape(x) for x in report.get("pipeline", []))

    css = """
    :root { --bg:#0f1419; --card:#1a2330; --text:#e8eef6; --muted:#93a4b8; --acc:#5eead4; --warn:#fbbf24; }
    body { margin:0; font-family:"Noto Sans SC","PingFang SC",sans-serif; background:var(--bg); color:var(--text); }
    header { padding:32px 40px 8px; }
    h1 { margin:0 0 8px; font-size:28px; }
    .sub { color:var(--muted); max-width:900px; line-height:1.5; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; padding:16px 40px 48px; }
    .card { background:var(--card); border-radius:14px; padding:18px 20px; box-shadow:0 10px 30px #0006; }
    h2 { margin:0 0 12px; font-size:16px; color:var(--acc); letter-spacing:.04em; }
    table.kv { width:100%; border-collapse:collapse; }
    table.kv th { text-align:left; color:var(--muted); font-weight:500; padding:6px 8px 6px 0; width:48%; }
    table.kv td { padding:6px 0; }
    ul { margin:8px 0 0; padding-left:18px; color:#d5e0ec; }
    li { margin:6px 0; }
    .pipe { font-size:13px; color:var(--muted); line-height:1.6; }
    .kpi { font-size:32px; font-weight:700; }
    .warn { color:var(--warn); }
    code { background:#0b1220; padding:1px 6px; border-radius:6px; }
    """
    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(report['title'])}</title>
<style>{css}</style></head><body>
<header>
  <h1>{html.escape(report['title'])}</h1>
  <p class="sub">用 origin.srt vs asr.srt 走一遍真实字幕 ASR 的评估闭环：文本 CER、时间轴 IoU、方案A MeanErr、语言瓶颈热词、幻觉后处理、自适应路由。这是报表页的学习版，对应生产 Gateway 后面的质量报告。</p>
  <p class="pipe">{pipeline}</p>
</header>
<section class="grid">
  <article class="card"><h2>文本 CER</h2>
    <div class="kpi warn">{pct(tm['cer_strip_punct'])}</div>
    {table([
        ("去标点 CER", pct(tm["cer_strip_punct"])),
        ("含标点 CER", pct(tm["cer_keep_punct"])),
        ("替换 / 插入 / 删除", f"{tm['substitutions']} / {tm['insertions']} / {tm['deletions']}"),
        ("ref 字数 / hyp 字数", f"{tm['ref_len']} / {tm['hyp_len']}"),
    ])}
  </article>
  <article class="card"><h2>时间轴对齐</h2>
    {table([
        ("Precision", pct(al["precision"])),
        ("Recall", pct(al["recall"])),
        ("origin 段 / ASR 段", f"{al['ref_segments']} / {al['hyp_segments']}"),
        ("未匹配 origin / ASR", f"{al['unmatched_ref']} / {al['unmatched_hyp']}"),
        ("起始偏移中位数", f"{al.get('start_offset_ms_median', 0):.0f} ms"),
    ])}
  </article>
  <article class="card"><h2>方案 A / A2 FA</h2>
    {table([
        ("MeanErr", f"{sa.get('mean_err_ms') or 0:.0f} ms"),
        ("< 800ms", "是" if sa.get("pass_mean_err_800ms") else "否"),
        ("覆盖率", pct(sa.get("coverage") or 0)),
        ("对齐字符", str(sa.get("aligned_chars"))),
    ])}
  </article>
  <article class="card"><h2>自适应路由</h2>
    {table([
        ("route", route["route"]),
        ("对白密度", f"{route['speech_density']:.2f}"),
        ("平均 CPS", f"{route['cue_cps_mean']:.2f}"),
        ("长间隙比", f"{route['long_gap_ratio']:.2f}"),
    ])}
    <p class="sub">{html.escape(route['reason'])}</p>
  </article>
  <article class="card"><h2>怎么读这份报告</h2><ul>{readout}</ul></article>
  <article class="card"><h2>语言瓶颈 · 替换</h2><ul>{subst or "<li>无</li>"}</ul></article>
  <article class="card"><h2>语言瓶颈 · 热词候选</h2><ul>{hotwords or "<li>无</li>"}</ul></article>
  <article class="card"><h2>轴切错（字在 origin 里）</h2><ul>{timing or "<li>无</li>"}</ul></article>
  <article class="card"><h2>幻觉候选</h2><ul>{hallu or "<li>无</li>"}</ul></article>
</section>
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out
