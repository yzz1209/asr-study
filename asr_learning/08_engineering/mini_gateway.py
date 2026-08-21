#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from asr_learning.lib.gateway import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="学习版 Gateway：报表页 + JSON API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = serve(args.host, args.port)
    print(json.dumps({"serving": f"http://{args.host}:{args.port}/report", "api": "/api/v1/asr/report"}, ensure_ascii=False))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
