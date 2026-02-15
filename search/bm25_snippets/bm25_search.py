from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
SYNONYMS = {
    "robotics": ["robot"],
    "coding": ["python"],
    "sensor": ["sensors"],
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_documents(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snippet(text: str, query_tokens: list[str], width: int = 7) -> str:
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-z0-9']", "", word.lower())
        if clean in query_tokens:
            start = max(0, i - width)
            end = min(len(words), i + width + 1)
            return " ".join(words[start:end])
    return " ".join(words[: min(len(words), width * 2)])


def search(
    documents: list[dict[str, str]], query: str, top_k: int = 3, min_score: float = 0.0
) -> list[tuple[str, float, str]]:
    query_tokens = tokenize(query)
    expanded = list(query_tokens)
    for token in query_tokens:
        expanded.extend(SYNONYMS.get(token, []))
    query_tokens = expanded
    tokenized = [tokenize(d["text"]) for d in documents]
    doc_count = len(documents)
    avg_len = sum(len(t) for t in tokenized) / max(1, doc_count)

    df: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    scored: list[tuple[str, float, str]] = []
    for doc, tokens in zip(documents, tokenized):
        counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (doc_count - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(1e-9, avg_len))
            score += idf * ((tf * 2.5) / denom)
        snippet = _snippet(doc["text"], query_tokens)
        scored.append((doc["title"], score, snippet))

    scored.sort(key=lambda row: row[1], reverse=True)
    return [row for row in scored if row[1] >= min_score][:top_k]


def main() -> None:
    base = Path(__file__).resolve().parent
    docs = load_documents(base / "documents.json")
    query = input("Enter search query: ").strip()
    top_k_text = input("Top K results [3]: ").strip()
    top_k = int(top_k_text) if top_k_text else 3
    for rank, (title, score, snippet) in enumerate(search(docs, query, top_k=top_k), start=1):
        print(f"{rank}. {title} | score={score:.4f}")
        print(f"   snippet: {snippet}")


if __name__ == "__main__":
    main()
