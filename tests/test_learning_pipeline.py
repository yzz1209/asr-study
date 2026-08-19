from asr_learning.lib.ctc import ctc_forced_alignment, ctc_greedy_decode
from asr_learning.lib.metrics import cer
from asr_learning.lib.srt import dump_srt, parse_srt
from asr_learning.lib.text_align import explode_chars, scheme_a_report
from asr_learning.lib.time_align import diagnose_alignment, interval_iou
from asr_learning.lib.hallucination import detect_hallucinations, ngram_repeat_score
from asr_learning.lib.vad import energy_vad_wav, write_sine_wav
from asr_learning.lib.adaptive import guess_route
from asr_learning.lib.features import hz_to_mel, mel_filterbank
from asr_learning.lib.mock_decoder import DecodeConfig, mock_decode
from asr_learning.lib.report import build_report
from asr_learning.lib.paths import ASR_SAMPLE, ORIGIN_SAMPLE
from asr_learning.lib.gateway import serve_in_thread
from asr_learning.lib.srt import load_srt
from asr_learning.lib.text import normalize_for_cer

import json
import urllib.request


def test_cer_known_pair():
    r = cer("我爱你", "我喜欢你", keep_punct=False)
    assert r["ref_len"] == 3
    assert r["distance"] >= 2
    assert 0 < r["cer"] <= 1


def test_normalize_punct_changes_cer():
    ref = "你好，世界"
    hyp = "你好世界"
    keep = cer(ref, hyp, keep_punct=True)
    strip = cer(ref, hyp, keep_punct=False)
    assert keep["cer"] > strip["cer"]
    assert strip["cer"] == 0
    assert normalize_for_cer("ＡＢＣ") == "ABC"


def test_srt_roundtrip():
    src = "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n2\n00:00:03,000 --> 00:00:04,500\n大荒\n"
    segs = parse_srt(src)
    assert len(segs) == 2
    back = dump_srt(segs)
    segs2 = parse_srt(back)
    assert segs2[0].text == "你好"
    assert segs2[1].start_ms == 3000


def test_iou_and_diagnosis(tmp_path=None):
    assert interval_iou(0, 10, 0, 10) == 1.0
    assert interval_iou(0, 10, 20, 30) == 0.0
    origin = load_srt(str(ORIGIN_SAMPLE))
    asr = load_srt(str(ASR_SAMPLE))
    d = diagnose_alignment(origin, asr, min_iou=0.3)
    assert d["ref_segments"] == 223
    assert d["hyp_segments"] == 116
    assert d["recall"] < 0.6
    assert d["precision"] > 0.8
    assert any("漏段" in m for m in d["failure_modes"])


def test_scheme_a_mean_err():
    origin = load_srt(str(ORIGIN_SAMPLE))
    asr = load_srt(str(ASR_SAMPLE))
    report = scheme_a_report(origin, asr)
    assert report["aligned_chars"] > 100
    assert report["mean_err_ms"] is not None
    assert explode_chars(asr)


def test_ctc_greedy_and_fa():
    frames = ["<blank>", "灵", "灵", "相", "<blank>", "师", "师", "<blank>"]
    assert ctc_greedy_decode(frames) == "灵相师"
    fa = ctc_forced_alignment("灵相师", frames)
    assert [c for c, _, _ in fa] == ["灵", "相", "师"]


def test_vad_demo(tmp_path):
    wav = tmp_path / "x.wav"
    write_sine_wav(wav, [(0.4, 1.1, 220.0), (1.6, 2.4, 330.0)])
    segs = energy_vad_wav(str(wav), rms_threshold=0.02, min_speech_ms=150, min_silence_ms=150)
    assert len(segs) == 2
    assert segs[0].start_ms < 600
    assert segs[1].start_ms > 1200


def test_hallucination_boh_and_overlap():
    origin = load_srt(str(ORIGIN_SAMPLE))
    asr = load_srt(str(ASR_SAMPLE))
    hits = detect_hallucinations(asr, origin)
    assert len(hits) >= 1
    assert ngram_repeat_score("哈哈哈哈哈哈") > 0.5


def test_mock_decoder_temperature_adds_hallucination():
    ref = "只要跟哥在一起去哪里都行"
    greedy = mock_decode(ref, DecodeConfig(0.0, 5, False), seed=1)
    hot = mock_decode(ref, DecodeConfig(1.0, 5, True), seed=1)
    assert hot.hallucination_spans >= greedy.hallucination_spans
    assert hot.latency_units > greedy.latency_units
    assert 0 <= greedy.no_speech_prob <= 1


def test_mel_bank():
    assert hz_to_mel(700) > 0
    fb = mel_filterbank(512, 16000, n_mels=20)
    assert len(fb) == 20
    assert any(any(x > 0 for x in row) for row in fb)


def test_hotwords_capture_domain_terms():
    from asr_learning.lib.hotwords import extract_hotwords
    from asr_learning.lib.taxonomy import hotword_candidates

    words = extract_hotwords(str(ORIGIN_SAMPLE), str(ASR_SAMPLE), limit=80)
    blob = "".join(words)
    assert len(words) >= 10
    assert "十二灵宫" in words or "灵相化形" in words or "飞花剑影" in blob
    origin = load_srt(str(ORIGIN_SAMPLE))
    asr = load_srt(str(ASR_SAMPLE))
    assert hotword_candidates(origin, asr)


def test_adaptive_and_full_report():
    origin = load_srt(str(ORIGIN_SAMPLE))
    asr = load_srt(str(ASR_SAMPLE))
    g = guess_route(origin, asr)
    assert g.route in {"belle_only", "paraformer_plus_fa", "belle_aggressive_filter"}
    report = build_report()
    assert report["text_metrics"]["cer_strip_punct"] > 0.3
    assert report["alignment"]["recall"] < 0.6
    assert "language_bottleneck" in report
    assert report["scheme_a"]["mean_err_ms"] is not None


def test_gateway_health():
    httpd, _ = serve_in_thread("127.0.0.1", 8768)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8768/api/v1/health", timeout=3) as resp:
            body = json.loads(resp.read().decode())
        assert body["ok"] is True
        with urllib.request.urlopen("http://127.0.0.1:8768/api/v1/asr/report", timeout=10) as resp:
            report = json.loads(resp.read().decode())
        assert "text_metrics" in report
    finally:
        httpd.shutdown()
