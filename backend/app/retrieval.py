"""Embeddings, vector search (Chroma), keyword search (BM25), hybrid fusion and access control.

Access control is enforced in two places so no unauthorized chunk can reach the LLM:
  * Chroma queries carry a `where` filter on the `department` metadata.
  * The BM25 candidate set is built only from chunks in the caller's departments.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Chunk, Document

COLLECTION_NAME = "chunks"
RRF_K = 60
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Sentence-Transformers wrapper. Uses encode_document/encode_query when the model provides them,
    so model-specific task prompts (e.g. for EmbeddingGemma) are applied correctly."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        encode = getattr(self._model, "encode_document", self._model.encode)
        return encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        encode = getattr(self._model, "encode_query", self._model.encode)
        return encode([text], normalize_embeddings=True)[0].tolist()


class HashEmbedder:
    """Deterministic, offline bag-of-words hashing embedder for tests and demos. Not a semantic model."""

    dimension = 384

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)


@lru_cache
def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=get_settings().chroma_path)
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


@dataclass
class Hit:
    key: str
    document_id: int
    title: str
    department: str
    page: int
    text: str
    score: float


def chunk_key(document_id: int, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def index_chunks(document: Document, chunks: list, embedder: Embedder | None = None) -> None:
    """Embed and store chunks in Chroma. `chunks` are TextChunk objects from ingestion."""
    if not chunks:
        return
    embedder = embedder or get_embedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    get_collection().add(
        ids=[chunk_key(document.id, c.index) for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "document_id": document.id,
                "title": document.title,
                "department": document.department,
                "page": c.page,
                "chunk_index": c.index,
            }
            for c in chunks
        ],
    )


def delete_document_vectors(document_id: int) -> None:
    get_collection().delete(where={"document_id": document_id})


def _vector_search(query: str, departments: list[str], k: int, embedder: Embedder) -> list[Hit]:
    collection = get_collection()
    if not departments or collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=min(k, collection.count()),
        where={"department": {"$in": departments}},
    )
    hits = []
    for key, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
    ):
        hits.append(
            Hit(
                key=key,
                document_id=meta["document_id"],
                title=meta["title"],
                department=meta["department"],
                page=meta["page"],
                text=doc,
                score=1.0 - dist,
            )
        )
    return hits


def _keyword_search(query: str, departments: list[str], k: int, session: Session) -> list[Hit]:
    if not departments:
        return []
    rows = session.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.department.in_(departments), Document.status == "indexed")
    ).all()
    if not rows:
        return []
    corpus = [_TOKEN_RE.findall(chunk.text.lower()) for chunk, _ in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_TOKEN_RE.findall(query.lower()))
    ranked = sorted(zip(scores, rows, strict=True), key=lambda item: item[0], reverse=True)[:k]
    hits = []
    for score, (chunk, doc) in ranked:
        if score <= 0:
            continue
        hits.append(
            Hit(
                key=chunk_key(chunk.document_id, chunk.chunk_index),
                document_id=chunk.document_id,
                title=doc.title,
                department=chunk.department,
                page=chunk.page,
                text=chunk.text,
                score=float(score),
            )
        )
    return hits


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Fuse several ranked lists. Each hit's score becomes its fused RRF score."""
    fused: dict[str, float] = {}
    by_key: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.key] = fused.get(hit.key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(hit.key, hit)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [Hit(**{**by_key[key].__dict__, "score": score}) for key, score in ordered]


def search(
    query: str,
    departments: list[str],
    session: Session,
    mode: str = "hybrid",
    top_k: int | None = None,
    embedder: Embedder | None = None,
) -> list[Hit]:
    """Return the top chunks the caller is allowed to see.

    mode: "semantic" (vector only) or "hybrid" (vector + BM25 fused with RRF).
    """
    settings = get_settings()
    k = top_k or settings.top_k
    embedder = embedder or get_embedder()
    candidates = settings.candidate_k
    vector_hits = _vector_search(query, departments, candidates, embedder)
    if mode == "semantic":
        return vector_hits[:k]
    keyword_hits = _keyword_search(query, departments, candidates, session)
    return reciprocal_rank_fusion([vector_hits, keyword_hits])[:k]
