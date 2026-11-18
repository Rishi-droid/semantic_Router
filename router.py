"""Semantic routing core: embeddings -> similarity -> route decision."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

FALLBACK = "fallback"
LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embedder(kind: str, api_key: str | None = None):
    """Return a LangChain Embeddings object. kind: 'local' or 'openai'."""
    if kind == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    raise ValueError(f"Unknown embedder kind: {kind!r}")


def _norm(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


@dataclass
class RouteResult:
    route: str  # final decision (may be FALLBACK)
    best_route: str  # closest route, even if below threshold
    scores: dict[str, float]
    latency_ms: float


class SemanticRouter:
    MODES = ("max", "centroid")

    def __init__(self, embedder, routes: dict[str, list[str]]):
        self.embedder = embedder
        self.names = list(routes)
        labels, texts = [], []
        for name, utterances in routes.items():
            if not utterances:
                raise ValueError(f"Route '{name}' has no example utterances")
            for u in utterances:
                labels.append(name)
                texts.append(u)
        self.labels = np.array(labels)
        self.examples = _norm(embedder.embed_documents(texts))
        self.centroids = _norm(
            np.stack([self.examples[self.labels == n].mean(axis=0) for n in self.names])
        )

    # -- embedding -----------------------------------------------------
    def embed_timed(self, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Embed queries one by one (realistic per-query latency)."""
        vectors, latencies = [], []
        for q in queries:
            t0 = time.perf_counter()
            vectors.append(self.embedder.embed_query(q))
            latencies.append((time.perf_counter() - t0) * 1000)
        return _norm(vectors), np.array(latencies)

    # -- scoring -------------------------------------------------------
    def scores(self, Q: np.ndarray, mode: str = "max") -> np.ndarray:
        """Return an (n_queries, n_routes) similarity matrix.

        max      -> similarity to the closest example utterance of each route
        centroid -> similarity to the mean embedding of each route
        """
        if mode == "centroid":
            return Q @ self.centroids.T
        sims = Q @ self.examples.T
        return np.stack(
            [sims[:, self.labels == n].max(axis=1) for n in self.names], axis=1
        )

    # -- routing -------------------------------------------------------
    def route(self, query: str, threshold: float, mode: str = "max") -> RouteResult:
        Q, lat = self.embed_timed([query])
        s = self.scores(Q, mode)[0]
        scores = dict(zip(self.names, map(float, s)))
        best = max(scores, key=scores.get)
        decision = best if scores[best] >= threshold else FALLBACK
        return RouteResult(decision, best, scores, float(lat[0]))