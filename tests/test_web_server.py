from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from web_server import DashboardHandler


class DashboardServerTests(unittest.TestCase):
    def test_server_import_does_not_require_streamlit(self) -> None:
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "streamlit" or name.startswith("streamlit."):
        raise ModuleNotFoundError("streamlit is not installed")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import web_server
import scripts.dashboard
"""
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )

    def test_serves_replay_and_only_known_assets(self) -> None:
        payload = json.dumps({"threshold": 0.9, "series": []}).encode()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(DashboardHandler, payload=payload)
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        root = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(f"{root}/api/replay") as response:
                self.assertEqual(json.load(response)["threshold"], 0.9)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            for path, content_type in (
                ("/", "text/html"),
                ("/app.js", "text/javascript"),
                ("/style.css", "text/css"),
                ("/plotly.min.js", "text/javascript"),
            ):
                with urlopen(f"{root}{path}") as response:
                    self.assertTrue(response.headers["Content-Type"].startswith(content_type))
                    self.assertGreater(len(response.read()), 0)
            with self.assertRaises(HTTPError) as error:
                urlopen(f"{root}/backend.py")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
