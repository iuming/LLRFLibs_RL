###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tiny zero-dependency local HTTP live view for rendered LLRF frames.

``HttpStreamer`` runs a localhost HTTP server that serves the latest rendered
frame as a PNG, plus a web page that auto-refreshes the image every ~150 ms.
This is the *robust* live channel: it works on machines where a Tk/Qt window
cannot be shown (headless sessions, remote terminals, CI-like environments),
because the operator just opens the URL in any browser::

    streamer = HttpStreamer(port=0).start()   # 0 -> ephemeral port
    streamer.publish(png_bytes)               # feed new frames
    print(streamer.url)                       # e.g. http://127.0.0.1:54321/
    ...
    streamer.close()

Only the standard library is used (``http.server`` + ``threading``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["HttpStreamer"]

# 1x1 transparent PNG used before the first real frame arrives.
_PLACEHOLDER = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

_HTML = (
    b"<!doctype html><html><head><meta charset='utf-8'>"
    b"<title>LLRF live console</title>"
    b"<style>body{background:#0d1117;color:#e6edf3;font-family:Consolas,monospace;"
    b"text-align:center} h3{color:#58a6ff} img{max-width:98%;border:1px solid #30363d}"
    b"</style></head><body><h3>LLRF live console</h3>"
    b"<img id='f' src='/frame' alt='live' width='960'>"
    b"<script>setInterval(function(){"
    b"document.getElementById('f').src='/frame?q='+Date.now();},150);"
    b"</script></body></html>"
)


class HttpStreamer:
    """Serve live-rendered PNG frames over localhost."""

    def __init__(self, port: int = 0, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._frame = _PLACEHOLDER
        self._count = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> HttpStreamer:
        """Bind the server on daemon threads (never blocks the caller)."""
        if self._server is None:
            self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
            self.port = int(self._server.server_address[1])
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """Shut the server down (idempotent)."""
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            finally:
                self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        """The URL to open in a browser."""
        return f"http://{self.host}:{self.port}/"

    # -- frame publishing -----------------------------------------------------

    def publish(self, png_bytes: bytes) -> None:
        """Publish the newest PNG frame (thread-safe)."""
        with self._lock:
            self._frame = png_bytes
            self._count += 1

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count

    def _snapshot(self) -> bytes:
        with self._lock:
            return self._frame

    # -- HTTP plumbing ----------------------------------------------------------

    def _make_handler(self) -> Callable:
        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self._send_html()
                elif self.path.startswith("/frame"):
                    self._send_png()
                else:
                    self.send_error(404)

            def _send_html(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_HTML)))
                self.end_headers()
                self.wfile.write(_HTML)

            def _send_png(self) -> None:
                data = streamer._snapshot()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:  # silence request logging
                pass

        return _Handler
