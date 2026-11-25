#!/usr/bin/env python3
"""Serve the local, browser-driven IMS demonstration."""

from __future__ import annotations

import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from plotly.offline import get_plotlyjs

from backend import BEARING_ORDER, load_dashboard_data


WEB_DIR = Path(__file__).resolve().parent / "web"


def build_payload() -> dict:
    data = load_dashboard_data()
    return {
        "threshold": data.selected_threshold,
        "activeModel": data.active_model,
        "modelThresholds": data.model_thresholds,
        "models": data.model_metrics,
        "metrics": data.model_metrics,
        "series": [
            {
                "id": bearing_id,
                "timestamps": series.timestamps.tolist(),
                "fileIndices": series.file_indices.tolist(),
                "probabilities": series.probabilities.tolist(),
                "probabilitiesByModel": {
                    model_id: probabilities.tolist()
                    for model_id, probabilities in series.probabilities_by_model.items()
                },
                "rms": series.rms.tolist(),
                "kurtosis": series.kurtosis.tolist(),
                "labels": series.labels.tolist(),
            }
            for bearing_id in BEARING_ORDER
            for series in (data.series_by_bearing[bearing_id],)
        ],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, payload: bytes, **kwargs):
        self.payload = payload
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path == "/api/replay":
            content, content_type = self.payload, "application/json; charset=utf-8"
        elif self.path == "/plotly.min.js":
            content, content_type = get_plotlyjs().encode("utf-8"), "text/javascript; charset=utf-8"
        elif self.path in files:
            filename, content_type = files[self.path]
            try:
                content = (WEB_DIR / filename).read_bytes()
            except OSError as error:
                self.log_error("Cannot read dashboard asset %s: %s", filename, error)
                self.send_error(500, "Dashboard asset unavailable")
                return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store" if self.path == "/api/replay" else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    payload = json.dumps(build_payload(), allow_nan=False).encode("utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 8502), partial(DashboardHandler, payload=payload)
    )
    print("StoneCut demo: http://127.0.0.1:8502", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
