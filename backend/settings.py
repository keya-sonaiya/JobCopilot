from functools import lru_cache
import os
from pathlib import Path
import sys
import warnings

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from pydantic import BaseModel


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# `gemma3:27b` was retired from Ollama Cloud in July 2026.
# Keep the default aligned with a model currently offered by the Cloud API.
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def configure_runtime() -> None:
    warnings.filterwarnings(
        "ignore", category=LangChainPendingDeprecationWarning)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def get_env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def get_env_int(name: str, default: int) -> int:
    value = get_env_value(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(name: str, default: float) -> float:
    value = get_env_value(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    value = get_env_value(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def normalize_ollama_host(host: str) -> str:
    normalized = host.rstrip("/")

    # The Ollama Python client expects the host root and appends /api itself.
    if normalized.endswith("/api"):
        normalized = normalized[: -len("/api")]

    if normalized == "https://api.ollama.com":
        return "https://ollama.com"

    return normalized


def is_remote_ollama_host(host: str) -> bool:
    return not host.startswith(("http://localhost", "http://127.0.0.1", "http://0.0.0.0"))


class AppSettings(BaseModel):
    ollama_api_key: str
    ollama_host: str
    ollama_model: str
    cors_origins: list[str]
    rag_retrieval_mode: str
    rag_fusion_method: str
    rag_fusion_alpha: float
    rag_embedding_provider: str
    rag_embedding_model: str
    rag_top_k: int
    rag_rrf_k: int
    rag_semantic_pool_factor: int
    rag_use_faiss: bool


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(BACKEND_DIR / ".env", override=True)

    cors_origins = [
        origin.strip()
        for origin in get_env_value("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]

    return AppSettings(
        ollama_api_key=get_env_value("OLLAMA_API_KEY"),
        ollama_host=normalize_ollama_host(
            get_env_value("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)),
        ollama_model=get_env_value("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        cors_origins=cors_origins,
        rag_retrieval_mode=get_env_value("RAG_RETRIEVAL_MODE", "hybrid"),
        rag_fusion_method=get_env_value("RAG_FUSION_METHOD", "rrf"),
        rag_fusion_alpha=get_env_float("RAG_FUSION_ALPHA", 0.65),
        rag_embedding_provider=get_env_value(
            "RAG_EMBEDDING_PROVIDER", "sentence-transformers"),
        rag_embedding_model=get_env_value(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rag_top_k=get_env_int("RAG_TOP_K", 6),
        rag_rrf_k=get_env_int("RAG_RRF_K", 60),
        rag_semantic_pool_factor=get_env_int("RAG_SEMANTIC_POOL_FACTOR", 3),
        rag_use_faiss=get_env_bool("RAG_USE_FAISS", True),
    )
