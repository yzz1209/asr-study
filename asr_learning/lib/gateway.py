from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import json
import threading

from .report import build_report, write_html, write_json


class ReportHandler(BaseHTTPRequestHandler):
    """Miniature of the production ASR Gateway + report page."""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/report"):
            report = build_report()
            html_path = write_html(report)
            self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/v1/health":
            self._send(200, json.dumps({"ok": True, "service": "asr-learning-gateway"}).encode(), "application/json")
            return
        if path == "/api/v1/asr/report":
            report = build_report()
            write_json(report)
            self._send(200, json.dumps(report, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/v1/asr/report":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        origin = payload.get("origin_srt_path")
        asr = payload.get("asr_srt_path")
        report = build_report(origin, asr)
        write_json(report)
        write_html(report)
        self._send(200, json.dumps(report, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), ReportHandler)
    return httpd


def serve_in_thread(host: str = "127.0.0.1", port: int = 8765) -> tuple[ThreadingHTTPServer, threading.Thread]:
    httpd = serve(host, port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t
