from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from textwrap import shorten

from resume_rag import ResumeRAGConfig, ResumeRAGIndex


SAMPLE_RESUME = """
Sam Rivera
Senior Machine Learning Engineer

Built production retrieval-augmented generation systems for support automation using Python,
FastAPI, LangChain, vector search, and evaluation dashboards. Improved answer grounding by
reducing hallucinated responses by 38%.

Led MLOps platform work on model deployment, CI/CD, feature stores, monitoring, and incident
response for real-time recommendation models serving millions of requests per day.

Earlier experience includes backend API development with PostgreSQL, Redis, Docker, Kubernetes,
and cloud infrastructure on AWS.
"""

DEFAULT_QUERIES = [
    "retrieval augmented generation and vector search experience",
    "MLOps deployment monitoring CI/CD",
    "backend APIs PostgreSQL Docker Kubernetes",
]


def load_text(path: Path | None) -> str:
    if not path:
        return SAMPLE_RESUME.strip()
    return path.read_text(encoding="utf-8").strip()


def load_queries(path: Path | None, inline_queries: list[str]) -> list[str]:
    if inline_queries:
        return [query.strip() for query in inline_queries if query.strip()]
    if not path:
        return DEFAULT_QUERIES

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return DEFAULT_QUERIES

    if path.suffix.lower() == ".json":
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("Query JSON must be a list of strings.")
        return [str(item).strip() for item in value if str(item).strip()]

    return [line.strip() for line in raw.splitlines() if line.strip()]


def print_result_block(mode: str, query: str, index: ResumeRAGIndex, top_k: int) -> None:
    print(f"\n[{mode}]")
    try:
        results = index.retrieve(query, top_k=top_k, mode=mode)
    except Exception as exc:
        print(f"  retrieval failed: {exc}")
        return

    if not results:
        print("  no chunks returned")
        return

    for rank, result in enumerate(results, start=1):
        token_score = f", token={result.token_score:.4f}" if result.token_score is not None else ""
        semantic_score = (
            f", semantic={result.semantic_score:.4f}" if result.semantic_score is not None else ""
        )
        metadata = result.metadata
        source = metadata.get("source") or "resume"
        doc_id = metadata.get("doc_id") or "resume"
        page = f", page={metadata['page']}" if metadata.get("page") is not None else ""
        fusion = f", fusion={metadata['fusion_method']}" if metadata.get("fusion_method") else ""
        vector_store = f", vector_store={metadata['vector_store']}" if metadata.get("vector_store") else ""
        snippet = shorten(" ".join(result.text.split()), width=220, placeholder="...")
        print(
            f"  {rank}. score={result.score:.4f}{token_score}{semantic_score}"
            f" | source={source}, doc_id={doc_id}, chunk={metadata.get('chunk_index', result.index)}"
            f"{page}{fusion}{vector_store}"
        )
        print(f"     {snippet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare token-only, semantic-only, and hybrid resume retrieval results."
    )
    parser.add_argument("--resume-file", type=Path, help="Plain-text resume file. Uses sample text when omitted.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Text file with one query per line, or a JSON list of query strings.",
    )
    parser.add_argument("--query", action="append", default=[], help="Inline query. Repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to show per mode.")
    parser.add_argument(
        "--fusion-method",
        choices=["rrf", "weighted"],
        default="rrf",
        help="Hybrid fusion strategy.",
    )
    parser.add_argument("--alpha", type=float, default=0.65, help="Semantic weight for weighted fusion.")
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "openai"],
        default="sentence-transformers",
        help="Embedding backend for semantic retrieval.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name. Use text-embedding-3-small for OpenAI.",
    )
    parser.add_argument("--no-faiss", action="store_true", help="Use exact cosine search instead of FAISS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_text = load_text(args.resume_file)
    queries = load_queries(args.queries_file, args.query)
    base_config = ResumeRAGConfig(
        retrieval_mode="hybrid",
        fusion_method=args.fusion_method,
        fusion_alpha=args.alpha,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        use_faiss=not args.no_faiss,
    ).normalized()

    print("Retrieval comparison")
    print(f"Chunks: {len(ResumeRAGIndex(resume_text, config=replace(base_config, retrieval_mode='token')).chunks)}")
    print(
        f"Semantic backend: {base_config.embedding_provider} / {base_config.embedding_model}; "
        f"fusion={base_config.fusion_method}, alpha={base_config.fusion_alpha}"
    )

    indexes: dict[str, ResumeRAGIndex | None] = {}
    for mode in ("token", "semantic", "hybrid"):
        config = replace(base_config, retrieval_mode=mode).normalized()
        try:
            indexes[mode] = ResumeRAGIndex(resume_text, config=config)
        except Exception as exc:
            indexes[mode] = None
            print(f"\n[{mode}] index build failed: {exc}")

    for query_number, query in enumerate(queries, start=1):
        print(f"\n=== Query {query_number}: {query} ===")
        for mode in ("token", "semantic", "hybrid"):
            index = indexes.get(mode)
            if index is None:
                print(f"\n[{mode}] unavailable")
                continue
            print_result_block(mode, query, index, args.top_k)


if __name__ == "__main__":
    main()
