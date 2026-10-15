"""HTTP endpoint for the ballot pipeline (stdlib only, no extra dependencies).

    python server.py                      # http://127.0.0.1:8000
    python server.py --host 0.0.0.0 --port 8000   # reachable from other machines

POST /evaluate[?first_number=N]  body: the image, either raw bytes or multipart/form-data (any file field)
    -> 200 {"output": int, "analysis": {..., "image": "<base64 PNG>"}}   (same object as ballot.evaluate)
    -> 400 {"error": "..."} if the body is not a readable image
GET  /health -> {"ok": true}
"""
import argparse
import base64
import json
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import ballot

MAX_BYTES = 20 * 1024 * 1024


def image_bytes(body, content_type):
    """Raw image body, or the first file part of a multipart/form-data body."""
    if content_type.startswith("multipart/form-data"):
        msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        for part in msg.iter_parts():
            if part.get_filename() or part.get_content_maintype() == "image":
                return part.get_payload(decode=True)
        return b""
    return body


def evaluate_json(data, first_number=1):
    """Image bytes -> JSON-ready evaluate() object (picture base64-encoded). Raises ValueError if unreadable."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise ValueError("body is not a readable image (send PNG/JPEG bytes or multipart form-data)")
    obj = ballot.evaluate(img, first_number)
    obj["analysis"]["image"] = base64.b64encode(obj["analysis"]["image"]).decode()
    return obj


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # browser front-ends on another origin
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "not found - POST an image to /evaluate"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/evaluate":
            return self.send_json(404, {"error": "not found - POST an image to /evaluate"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BYTES:
            return self.send_json(413, {"error": f"image larger than {MAX_BYTES // 2**20} MB"})
        try:
            first = int(parse_qs(url.query).get("first_number", ["1"])[0])
            data = image_bytes(self.rfile.read(size), self.headers.get("Content-Type", ""))
            self.send_json(200, evaluate_json(data, first))
        except ValueError as e:
            self.send_json(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="Ballot pipeline HTTP endpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"ballot pipeline on http://{a.host}:{a.port}  (POST /evaluate, GET /health)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
