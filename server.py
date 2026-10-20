"""Local chat page. It listens on this machine only."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from assistant.memory import redact
from assistant.present import present
from main import build

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"
HOST = "127.0.0.1"


class App:
    def __init__(self, live: bool):
        self.live = live
        self.lock = Lock()
        self.assistant = build(live)


def status(app: App) -> dict:
    index = app.assistant.index
    profile = app.assistant.memory.profile
    return {
        "live": app.live,
        "name": profile.get("name", ""),
        "customer_id": profile.get("customer_id", ""),
        "segment": profile.get("segment", ""),
        "retrieval": index.method,
        "embed_error": index.embed_error,
    }


def reply(app: App, message: str) -> dict:
    answer = app.assistant.ask(message, live=app.live)
    text = answer.text if app.live else present(message, answer)
    return {
        "echo": redact(message).strip(),
        "text": text,
        "path": answer.path,
        "retrieval": answer.retrieval,
        "sources": answer.sources,
        "audit": answer.audit,
    }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/status":
                with app.lock:
                    body = status(app)
                self._json(200, body)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                data = self._read_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "Geçersiz istek."})
                return
            if path == "/api/chat":
                message = str(data.get("message", ""))
                try:
                    with app.lock:
                        body = reply(app, message)
                except ValueError as error:
                    self._json(400, {"error": str(error)})
                    return
                self._json(200, body)
                return
            if path == "/api/reset":
                with app.lock:
                    if data.get("ledger"):
                        app.assistant = build(app.live)
                    else:
                        app.assistant.reset()
                    body = status(app)
                self._json(200, body)
                return
            self._json(404, {"error": "Yok."})

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _json(self, code: int, body: dict) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self._send(code, encoded, "application/json; charset=utf-8")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank Assistant sayfası")
    parser.add_argument("--live", action="store_true", help="Cevabı modelden al")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = App(live=args.live)
    server = ThreadingHTTPServer((HOST, args.port), make_handler(app))
    print(f"http://{HOST}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
