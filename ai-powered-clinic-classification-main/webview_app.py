import argparse
from pathlib import Path

import webview

from api import create_api

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_repo(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def main():
    p = argparse.ArgumentParser(description="Anemia classifier (PyWebView)")
    p.add_argument("--model", default="model/anemia_model.joblib")
    args = p.parse_args()

    model_path = resolve_from_repo(Path(args.model))
    api = create_api(model_path=model_path) # type: ignore

    index_path = (REPO_ROOT / "web" / "index.html").as_uri()

    webview.create_window(
        "Anemia classification",
        index_path,
        js_api=api,
        width=1200,
        height=800,
        min_size=(980, 680),
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()