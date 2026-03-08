from __future__ import annotations

import json
from pathlib import Path

from bm25_search import load_documents, search


def run(path: Path) -> list[dict[str, str]]:
    docs = load_documents(path / "documents.json")
    queries = [
        "robotics demos",
        "python sessions",
        "soil sensor trends",
    ]
    report = []
    for query in queries:
        top = search(docs, query, top_k=1)[0]
        report.append({"query": query, "top_title": top[0]})
    return report


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    print(json.dumps(run(base), indent=2))
