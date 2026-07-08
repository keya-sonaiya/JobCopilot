from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
VALID_RETRIEVAL_MODES = {"token", "semantic", "hybrid"}
VALID_FUSION_METHODS = {"rrf", "weighted"}

ChunkMetadata = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ResumeRAGConfig:
    retrieval_mode: str = "hybrid"
    fusion_method: str = "rrf"
    fusion_alpha: float = 0.65
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 6
    rrf_k: int = 60
    semantic_pool_factor: int = 3
    use_faiss: bool = True

    @classmethod
    def from_env(cls) -> "ResumeRAGConfig":
        return cls(
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", cls.retrieval_mode),
            fusion_method=os.getenv("RAG_FUSION_METHOD", cls.fusion_method),
            fusion_alpha=_env_float("RAG_FUSION_ALPHA", cls.fusion_alpha),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", cls.embedding_provider),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            top_k=_env_int("RAG_TOP_K", cls.top_k),
            rrf_k=_env_int("RAG_RRF_K", cls.rrf_k),
            semantic_pool_factor=_env_int("RAG_SEMANTIC_POOL_FACTOR", cls.semantic_pool_factor),
            use_faiss=_env_bool("RAG_USE_FAISS", cls.use_faiss),
        ).normalized()

    def normalized(self) -> "ResumeRAGConfig":
        retrieval_mode = self.retrieval_mode.strip().lower()
        if retrieval_mode not in VALID_RETRIEVAL_MODES:
            retrieval_mode = "hybrid"

        fusion_method = self.fusion_method.strip().lower()
        if fusion_method not in VALID_FUSION_METHODS:
            fusion_method = "rrf"

        provider = self.embedding_provider.strip().lower().replace("_", "-")
        if provider in {"sentence-transformer", "sentence-transformers", "local"}:
            provider = "sentence-transformers"
        elif provider in {"openai", "text-embedding-3-small"}:
            provider = "openai"

        embedding_model = self.embedding_model.strip()
        if provider == "openai" and (not embedding_model or embedding_model.startswith("sentence-transformers/")):
            embedding_model = "text-embedding-3-small"
        elif provider == "sentence-transformers" and (
            not embedding_model or embedding_model.startswith("text-embedding-")
        ):
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        return ResumeRAGConfig(
            retrieval_mode=retrieval_mode,
            fusion_method=fusion_method,
            fusion_alpha=min(max(self.fusion_alpha, 0.0), 1.0),
            embedding_provider=provider,
            embedding_model=embedding_model,
            top_k=max(self.top_k, 1),
            rrf_k=max(self.rrf_k, 1),
            semantic_pool_factor=max(self.semantic_pool_factor, 1),
            use_faiss=self.use_faiss,
        )


@dataclass(frozen=True)
class ResumeChunk:
    index: int
    text: str
    metadata: ChunkMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedResumeChunk:
    index: int
    score: float
    text: str
    metadata: ChunkMetadata = field(default_factory=dict)
    token_score: float | None = None
    semantic_score: float | None = None


class EmbeddingModel(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    name: str

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        ...


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 140) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph

    if current:
        chunks.append(current)

    if not chunks and text.strip():
        stride = max(max_chars - overlap_chars, 1)
        for start in range(0, len(text), stride):
            chunks.append(text[start : start + max_chars].strip())

    return [chunk for chunk in chunks if chunk]


def build_resume_chunks(
    resume_text: str,
    metadata: ChunkMetadata | None = None,
) -> list[ResumeChunk]:
    base_metadata = dict(metadata or {})
    source = base_metadata.get("source") or "resume"
    doc_id = base_metadata.get("doc_id") or "resume"

    chunks = []
    for index, text in enumerate(_chunk_text(resume_text)):
        chunk_metadata = {
            **base_metadata,
            "source": source,
            "doc_id": doc_id,
            "chunk_index": index,
        }
        chunks.append(ResumeChunk(index=index, text=text, metadata=chunk_metadata))
    return chunks


class TokenResumeRetriever:
    """TF-IDF-style token branch preserved from the original local retriever."""

    def __init__(self, chunks: Sequence[ResumeChunk]):
        self.chunks = list(chunks)
        self.chunk_tokens = [Counter(_tokens(chunk.text)) for chunk in self.chunks]
        self.document_frequency = Counter(
            token for chunk_counter in self.chunk_tokens for token in chunk_counter.keys()
        )
        self.total_chunks = max(len(self.chunks), 1)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedResumeChunk]:
        query_tokens = Counter(_tokens(query))
        if not query_tokens or not self.chunks:
            return []

        scored: list[RetrievedResumeChunk] = []
        for index, chunk_counter in enumerate(self.chunk_tokens):
            score = 0.0
            chunk_length = sum(chunk_counter.values()) or 1

            for token, query_count in query_tokens.items():
                frequency = chunk_counter.get(token, 0)
                if not frequency:
                    continue

                idf = math.log((self.total_chunks + 1) / (self.document_frequency[token] + 0.5)) + 1
                normalized_tf = frequency / chunk_length
                score += idf * query_count * normalized_tf

            if score:
                chunk = self.chunks[index]
                scored.append(
                    RetrievedResumeChunk(
                        index=chunk.index,
                        score=score,
                        text=chunk.text,
                        metadata={**chunk.metadata, "retrieval_mode": "token"},
                        token_score=score,
                    )
                )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class SentenceTransformersEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install backend requirements or set RAG_EMBEDDING_PROVIDER=openai."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class OpenAIEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = None

    def _load_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI semantic retrieval requires the openai package. "
                    "Install backend requirements and set OPENAI_API_KEY."
                ) from exc
            self._client = OpenAI()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._load_client().embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in response.data]


def create_embedding_model(config: ResumeRAGConfig) -> EmbeddingModel:
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingModel(config.embedding_model or "text-embedding-3-small")
    if config.embedding_provider == "sentence-transformers":
        return SentenceTransformersEmbeddingModel(config.embedding_model)
    raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {config.embedding_provider}")


def _unit_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


class ExactCosineVectorStore:
    name = "exact-cosine"

    def __init__(self, embeddings: Sequence[Sequence[float]]):
        self.embeddings = [_unit_vector(vector) for vector in embeddings]

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        query = _unit_vector(query_embedding)
        scored = []
        for index, embedding in enumerate(self.embeddings):
            score = sum(query_value * doc_value for query_value, doc_value in zip(query, embedding))
            scored.append((index, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class FaissVectorStore:
    name = "faiss"

    def __init__(self, embeddings: Sequence[Sequence[float]]):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("FAISS is not installed.") from exc

        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("FAISS index requires at least one embedding.")

        faiss.normalize_L2(matrix)
        self._faiss = faiss
        self._np = np
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        query = self._np.asarray([query_embedding], dtype="float32")
        self._faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0
        ]


def create_vector_store(
    embeddings: Sequence[Sequence[float]],
    use_faiss: bool = True,
) -> VectorStore:
    if use_faiss:
        try:
            return FaissVectorStore(embeddings)
        except RuntimeError:
            return ExactCosineVectorStore(embeddings)
    return ExactCosineVectorStore(embeddings)


class SemanticResumeRetriever:
    def __init__(
        self,
        chunks: Sequence[ResumeChunk],
        config: ResumeRAGConfig,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.chunks = list(chunks)
        self.config = config.normalized()
        self.embedding_model = embedding_model or create_embedding_model(self.config)
        self.vector_store: VectorStore | None = None

        if self.chunks:
            chunk_embeddings = self.embedding_model.embed([chunk.text for chunk in self.chunks])
            self.vector_store = create_vector_store(chunk_embeddings, use_faiss=self.config.use_faiss)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedResumeChunk]:
        if not query.strip() or not self.vector_store or not self.chunks:
            return []

        query_embedding = self.embedding_model.embed([query])[0]
        matches = self.vector_store.search(query_embedding, top_k=top_k)
        results = []
        for internal_index, score in matches:
            chunk = self.chunks[internal_index]
            results.append(
                RetrievedResumeChunk(
                    index=chunk.index,
                    score=score,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "retrieval_mode": "semantic",
                        "vector_store": self.vector_store.name,
                    },
                    semantic_score=score,
                )
            )
        return results


def _normalize_scores(results: Sequence[RetrievedResumeChunk]) -> dict[int, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {result.index: 1.0 for result in results}
    return {
        result.index: (result.score - min_score) / (max_score - min_score)
        for result in results
    }


class ResumeRAGIndex:
    def __init__(
        self,
        resume_text: str,
        config: ResumeRAGConfig | None = None,
        metadata: ChunkMetadata | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.config = (config or ResumeRAGConfig.from_env()).normalized()
        self.chunks = build_resume_chunks(resume_text, metadata=metadata)
        self.token_retriever = TokenResumeRetriever(self.chunks)
        self._semantic_retriever: SemanticResumeRetriever | None = None
        self._embedding_model = embedding_model

    def _get_semantic_retriever(self) -> SemanticResumeRetriever:
        if self._semantic_retriever is None:
            self._semantic_retriever = SemanticResumeRetriever(
                self.chunks,
                self.config,
                embedding_model=self._embedding_model,
            )
        return self._semantic_retriever

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        fusion_method: str | None = None,
        alpha: float | None = None,
    ) -> list[RetrievedResumeChunk]:
        top_k = top_k or self.config.top_k
        retrieval_mode = (mode or self.config.retrieval_mode).strip().lower()
        if retrieval_mode not in VALID_RETRIEVAL_MODES:
            retrieval_mode = self.config.retrieval_mode

        if retrieval_mode == "token":
            return self.token_retriever.retrieve(query, top_k=top_k)

        if retrieval_mode == "semantic":
            return self._get_semantic_retriever().retrieve(query, top_k=top_k)

        pool_k = max(top_k, top_k * self.config.semantic_pool_factor)
        token_results = self.token_retriever.retrieve(query, top_k=pool_k)
        try:
            semantic_results = self._get_semantic_retriever().retrieve(query, top_k=pool_k)
        except Exception:
            if token_results:
                return token_results[:top_k]
            raise

        method = (fusion_method or self.config.fusion_method).strip().lower()
        if method == "weighted":
            return self._weighted_score_fusion(
                token_results,
                semantic_results,
                top_k=top_k,
                alpha=self.config.fusion_alpha if alpha is None else alpha,
            )
        return self._reciprocal_rank_fusion(token_results, semantic_results, top_k=top_k)

    def _reciprocal_rank_fusion(
        self,
        token_results: Sequence[RetrievedResumeChunk],
        semantic_results: Sequence[RetrievedResumeChunk],
        top_k: int,
    ) -> list[RetrievedResumeChunk]:
        fused_scores: dict[int, float] = defaultdict(float)
        token_scores = {result.index: result.score for result in token_results}
        semantic_scores = {result.index: result.score for result in semantic_results}
        vector_store = next(
            (
                result.metadata.get("vector_store")
                for result in semantic_results
                if result.metadata.get("vector_store")
            ),
            None,
        )

        for results in (token_results, semantic_results):
            for rank, result in enumerate(results, start=1):
                fused_scores[result.index] += 1.0 / (self.config.rrf_k + rank)

        return self._build_fused_results(
            fused_scores,
            token_scores,
            semantic_scores,
            top_k=top_k,
            fusion_method="rrf",
            vector_store=vector_store,
        )

    def _weighted_score_fusion(
        self,
        token_results: Sequence[RetrievedResumeChunk],
        semantic_results: Sequence[RetrievedResumeChunk],
        top_k: int,
        alpha: float,
    ) -> list[RetrievedResumeChunk]:
        alpha = min(max(alpha, 0.0), 1.0)
        token_scores = {result.index: result.score for result in token_results}
        semantic_scores = {result.index: result.score for result in semantic_results}
        token_normalized = _normalize_scores(token_results)
        semantic_normalized = _normalize_scores(semantic_results)
        vector_store = next(
            (
                result.metadata.get("vector_store")
                for result in semantic_results
                if result.metadata.get("vector_store")
            ),
            None,
        )

        fused_scores = {}
        for index in set(token_normalized) | set(semantic_normalized):
            fused_scores[index] = (
                alpha * semantic_normalized.get(index, 0.0)
                + (1.0 - alpha) * token_normalized.get(index, 0.0)
            )

        return self._build_fused_results(
            fused_scores,
            token_scores,
            semantic_scores,
            top_k=top_k,
            fusion_method="weighted",
            vector_store=vector_store,
            extra_metadata={"fusion_alpha": round(alpha, 4)},
        )

    def _build_fused_results(
        self,
        fused_scores: dict[int, float],
        token_scores: dict[int, float],
        semantic_scores: dict[int, float],
        top_k: int,
        fusion_method: str,
        vector_store: str | int | float | bool | None = None,
        extra_metadata: ChunkMetadata | None = None,
    ) -> list[RetrievedResumeChunk]:
        results = []
        for index, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            chunk = self.chunks[index]
            metadata = {
                **chunk.metadata,
                "retrieval_mode": "hybrid",
                "fusion_method": fusion_method,
                **(extra_metadata or {}),
            }
            if vector_store:
                metadata["vector_store"] = vector_store
            results.append(
                RetrievedResumeChunk(
                    index=chunk.index,
                    score=score,
                    text=chunk.text,
                    metadata=metadata,
                    token_score=token_scores.get(index),
                    semantic_score=semantic_scores.get(index),
                )
            )
        return results


def _format_metadata(chunk: RetrievedResumeChunk) -> str:
    source = chunk.metadata.get("source") or "resume"
    doc_id = chunk.metadata.get("doc_id") or "resume"
    page = chunk.metadata.get("page")
    chunk_index = chunk.metadata.get("chunk_index", chunk.index)
    parts = [
        f"source={source}",
        f"doc_id={doc_id}",
        f"chunk={chunk_index}",
    ]
    if page is not None:
        parts.append(f"page={page}")
    if chunk.metadata.get("retrieval_mode"):
        parts.append(f"mode={chunk.metadata['retrieval_mode']}")
    if chunk.metadata.get("fusion_method"):
        parts.append(f"fusion={chunk.metadata['fusion_method']}")
    if chunk.metadata.get("vector_store"):
        parts.append(f"vector_store={chunk.metadata['vector_store']}")
    parts.append(f"score={chunk.score:.4f}")
    if chunk.token_score is not None:
        parts.append(f"token_score={chunk.token_score:.4f}")
    if chunk.semantic_score is not None:
        parts.append(f"semantic_score={chunk.semantic_score:.4f}")
    return ", ".join(parts)


def format_resume_rag_context(chunks: Sequence[RetrievedResumeChunk]) -> str:
    return "\n\n".join(
        f"[Resume excerpt {position}: {_format_metadata(chunk)}]\n{chunk.text}"
        for position, chunk in enumerate(chunks, start=1)
    )


def build_resume_rag_context(
    resume_text: str,
    queries: Iterable[str],
    top_k: int | None = None,
    config: ResumeRAGConfig | None = None,
    metadata: ChunkMetadata | None = None,
) -> str:
    query = "\n\n".join(query for query in queries if query and query.strip())
    index = ResumeRAGIndex(resume_text, config=config, metadata=metadata)
    chunks = index.retrieve(query, top_k=top_k)

    if not chunks:
        return resume_text[:4000]

    return format_resume_rag_context(chunks)
