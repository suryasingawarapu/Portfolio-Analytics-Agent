"""Retrieval-augmented few-shot examples for SQL generation.

The example bank (example_bank.json) is plain text source, committed to the repo like any
other file. Its embeddings are never written to disk: they're computed once, lazily, the
first time an example is retrieved, held in memory for the life of the process, and
discarded on exit. Every run re-embeds from scratch -- cheap enough (one batched API call
for the whole bank) that there's no need to cache it, and it keeps the repo free of
generated artifacts a reviewer would have to trust blindly.
"""
import json
from pathlib import Path

import numpy as np

from config import client, EMBEDDING_MODEL_NAME

EXAMPLE_BANK_FILE = Path(__file__).parent / "example_bank.json"

_examples: list[dict] | None = None
_embeddings: np.ndarray | None = None


def _embed(texts: list[str]) -> np.ndarray:
    """One batched call embeds every text at once -- cheaper than one call per example."""
    response = client.models.embed_content(model=EMBEDDING_MODEL_NAME, contents=texts)
    return np.array([e.values for e in response.embeddings])


def _ensure_loaded() -> None:
    global _examples, _embeddings
    if _examples is not None:
        return
    _examples = json.loads(EXAMPLE_BANK_FILE.read_text())
    _embeddings = _embed([e["question"] for e in _examples])


def retrieve(question: str, k: int = 3) -> list[dict]:
    """Return the k examples whose questions are most similar to the given one.

    Exact (brute-force) cosine-similarity nearest-neighbor search -- appropriate at this
    scale (a few dozen examples): no approximate index is worth the overhead.
    """
    _ensure_loaded()
    query_vector = _embed([question])[0]

    norms = np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(query_vector)
    similarities = (_embeddings @ query_vector) / norms

    top_k = np.argsort(similarities)[::-1][:k]
    return [_examples[i] for i in top_k]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What percent of the Balanced Portfolio is bonds?"
    for ex in retrieve(q):
        print(f"- {ex['question']}")
