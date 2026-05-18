import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from api import create_multi_model_api

REPO_ROOT = Path(__file__).resolve().parent
WEB_ROOT = REPO_ROOT / "web"
MODEL_CONFIGS = {
    "anemia": {
        "path": REPO_ROOT / "model" / "anemia_model.joblib",
        "title": "Anemia Classifier",
        "description": "Predicts anemia category from CBC and nutrition markers.",
        "supports_ocr": True,
    },
    "diabetes": {
        "path": REPO_ROOT / "model" / "diabetes_model_xgboost.joblib",
        "title": "Diabetes Status Classifier",
        "description": "Predicts diabetes risk (no diabetes vs diabetes) from health indicators.",
        "supports_ocr": True,
    },
    "leukemia": {
        "path": REPO_ROOT / "lekumiai_model",
        "title": "Leukemia Image Classifier",
        "description": "Detects suspected leukemia from microscopic blood smear images.",
        "supports_ocr": False,
        "type": "leukemia_image",
    },
}


def resolve_from_repo(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class AppHandler(BaseHTTPRequestHandler):
    api = None

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/api/model-info":
            model_key = qs.get("model", ["anemia"])[0]
            return self._json_response(HTTPStatus.OK, self.api.get_model_info(model_key))
        if parsed.path == "/api/models":
            return self._json_response(HTTPStatus.OK, self.api.list_models())
        if parsed.path == "/" or parsed.path == "":
            return self._serve_file(WEB_ROOT / "index.html")
        return self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            payload = self._read_json()
            model_key = payload.get("model", "anemia")
            sample = payload.get("sample", {})
            return self._json_response(HTTPStatus.OK, self.api.predict(model_key, sample))
        if parsed.path == "/api/predict-batch":
            payload = self._read_json()
            model_key = payload.get("model", "anemia")
            samples = payload.get("samples", [])
            return self._json_response(HTTPStatus.OK, self.api.predict_batch(model_key, samples))
        if parsed.path == "/api/extract-from-image":
            payload = self._read_json()
            model_key = payload.get("model", "anemia")
            b64 = payload.get("b64", "")
            return self._json_response(HTTPStatus.OK, self.api.extract_from_image(model_key, b64))
        if parsed.path == "/api/predict-image":
            payload = self._read_json()
            model_key = payload.get("model", "leukemia")
            b64 = payload.get("b64", "")
            return self._json_response(HTTPStatus.OK, self.api.predict_image(model_key, b64))
        return self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, fmt, *args):
        return

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        raise ValueError("Invalid JSON payload")

    def _json_response(self, status: HTTPStatus, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, raw_path: str):
        rel_path = raw_path.lstrip("/")
        candidate = (WEB_ROOT / rel_path).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            return self._json_response(HTTPStatus.FORBIDDEN, {"error": "Forbidden"})
        if candidate.is_dir():
            candidate = candidate / "index.html"
        return self._serve_file(candidate)

    def _serve_file(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            return self._json_response(HTTPStatus.NOT_FOUND, {"error": "File not found"})

        content = file_path.read_bytes()
        mime, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ValueError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Clinical classifier web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_paths = {
        key: {
            **cfg,
            "path": resolve_from_repo(Path(cfg["path"])),
        }
        for key, cfg in MODEL_CONFIGS.items()
    }
    AppHandler.api = create_multi_model_api(model_paths=model_paths)

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving HemoScan AI at {url}")
    print("Open that URL in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
