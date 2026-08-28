from functools import lru_cache
from copy import deepcopy
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4
import warnings
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from pydantic import BaseModel, Field, field_validator

from settings import configure_runtime, get_settings, is_remote_ollama_host
from documents import DocumentExtractionError, extract_resume_text
from resume_rag import ResumeRAGConfig


configure_runtime()

_showwarning = warnings.showwarning


def _hide_langchain_pending_warning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, LangChainPendingDeprecationWarning):
        return
    _showwarning(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _hide_langchain_pending_warning
from agent import ResumeJobApplicationSystem  # noqa: E402
warnings.showwarning = _showwarning

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_OUT_DIR = PROJECT_ROOT / "frontend" / "out"
FRONTEND_ASSETS_DIR = FRONTEND_OUT_DIR / "_next"

JOB_STEPS = [
    ("extract_resume", "Resume document"),
    ("parse_resume", "Resume parsing"),
    ("parse_job", "Job parsing"),
    ("company_search", "Company search"),
    ("resume_rag", "Resume RAG"),
    ("skill_match", "Skill matching"),
    ("cover_letter", "Cover letter"),
    ("recruiter_questions", "Follow-up chat"),
    ("finalize", "Finalize"),
]

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApplicationRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description_text: str = Field(..., min_length=20)
    recruiter_questions: list[str] = Field(default_factory=list)
    enable_company_search: bool = True

    @field_validator("resume_text", "job_description_text", mode="before")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("recruiter_questions", mode="before")
    @classmethod
    def normalize_questions(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class ApplicationResponse(BaseModel):
    status: str
    processing_time_seconds: float | None
    errors: list[str]
    summary: dict[str, Any]
    cover_letter: dict[str, Any] | None
    cover_letter_text: str | None
    recruiter_answers: list[dict[str, Any]]
    company_research: dict[str, Any] | None
    company_research_text: str | None
    resume_rag_context: str | None
    skill_match_analysis: dict[str, Any] | None
    parsed_resume: dict[str, Any] | None
    parsed_job_description: dict[str, Any] | None


class JobStartResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    steps: list[dict[str, Any]]
    result: ApplicationResponse | None = None
    error: str | None = None
    resume_file: dict[str, Any] | None = None
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
    latest_cover_letter: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    cover_letter_text: str | None = None
    enable_web_search: bool = False

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class ChatResponse(BaseModel):
    reply: str
    cover_letter_text: str | None
    cover_letter_updated: bool
    suggestions: list[str]
    messages: list[dict[str, Any]]
    web_search: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def get_application_system() -> ResumeJobApplicationSystem:
    settings = get_settings()
    rag_config = ResumeRAGConfig(
        retrieval_mode=settings.rag_retrieval_mode,
        fusion_method=settings.rag_fusion_method,
        fusion_alpha=settings.rag_fusion_alpha,
        embedding_provider=settings.rag_embedding_provider,
        embedding_model=settings.rag_embedding_model,
        top_k=settings.rag_top_k,
        rrf_k=settings.rag_rrf_k,
        semantic_pool_factor=settings.rag_semantic_pool_factor,
        use_faiss=settings.rag_use_faiss,
    ).normalized()

    if is_remote_ollama_host(settings.ollama_host) and not settings.ollama_api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is required for remote Ollama hosts. "
            "Set it in backend/.env or use OLLAMA_HOST=http://localhost:11434 for a local Ollama server."
        )

    return ResumeJobApplicationSystem(
        api_key=settings.ollama_api_key,
        host=settings.ollama_host,
        model=settings.ollama_model,
        rag_config=rag_config,
    )


def empty_steps() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": label,
            "status": "pending",
            "message": "",
            "updated_at": None,
        }
        for key, label in JOB_STEPS
    ]


def create_job(resume_file: dict[str, Any] | None = None) -> str:
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "steps": empty_steps(),
            "result": None,
            "error": None,
            "resume_file": resume_file,
            "chat_history": [],
            "latest_cover_letter": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    return job_id


def update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updated_at"] = utc_now()


def update_step(job_id: str, key: str, status: str, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for step in job["steps"]:
            if step["key"] == key:
                step["status"] = status
                step["message"] = message
                step["updated_at"] = utc_now()
                break
        job["updated_at"] = utc_now()


def build_application_response(result: Any) -> ApplicationResponse:
    return ApplicationResponse(
        status=result.status,
        processing_time_seconds=result.processing_time_seconds,
        errors=result.errors,
        summary=result.get_summary(),
        cover_letter=result.cover_letter.model_dump(mode="json") if result.cover_letter else None,
        cover_letter_text=result.cover_letter.get_full_letter() if result.cover_letter else None,
        recruiter_answers=[answer.model_dump(mode="json") for answer in result.recruiter_answers or []],
        company_research=result.company_research,
        company_research_text=result.company_research_text,
        resume_rag_context=result.resume_rag_context,
        skill_match_analysis=result.skill_match_analysis.model_dump(mode="json") if result.skill_match_analysis else None,
        parsed_resume=result.parsed_resume.model_dump(mode="json") if result.parsed_resume else None,
        parsed_job_description=result.parsed_job_description.model_dump(mode="json")
        if result.parsed_job_description
        else None,
    )


def run_application_job(
    job_id: str,
    resume_text: str,
    job_description_text: str,
    recruiter_questions: list[str],
    enable_company_search: bool,
    resume_file_metadata: dict[str, Any] | None = None,
) -> None:
    update_job(job_id, status="running")

    try:
        system = get_application_system()
        result = system.run_application_process(
            resume_text=resume_text,
            job_description_text=job_description_text,
            recruiter_questions=recruiter_questions,
            enable_company_search=enable_company_search,
            resume_metadata=resume_file_metadata,
            progress_callback=lambda key, status, message: update_step(job_id, key, status, message),
        )
        application_response = build_application_response(result).model_dump(mode="json")
        if result.errors:
            update_job(
                job_id,
                status="failed",
                error="; ".join(result.errors),
                result=application_response,
            )
            return

        update_job(
            job_id,
            status="completed",
            result=application_response,
            latest_cover_letter=application_response.get("cover_letter_text"),
        )
    except Exception as exc:
        update_step(job_id, "finalize", "failed", str(exc))
        update_job(job_id, status="failed", error=str(exc))


def build_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="JobCopilot API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.ollama_model,
            "host": settings.ollama_host,
            "rag": {
                "retrieval_mode": settings.rag_retrieval_mode,
                "fusion_method": settings.rag_fusion_method,
                "fusion_alpha": settings.rag_fusion_alpha,
                "embedding_provider": settings.rag_embedding_provider,
                "embedding_model": settings.rag_embedding_model,
                "top_k": settings.rag_top_k,
                "use_faiss": settings.rag_use_faiss,
            },
        }

    @app.post("/api/process", response_model=ApplicationResponse)
    async def process_application(payload: ApplicationRequest) -> ApplicationResponse:
        try:
            system = get_application_system()
            result = await run_in_threadpool(
                system.run_application_process,
                resume_text=payload.resume_text,
                job_description_text=payload.job_description_text,
                recruiter_questions=payload.recruiter_questions,
                enable_company_search=payload.enable_company_search,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ollama processing failed: {exc}") from exc

        return build_application_response(result)

    @app.post("/api/jobs", response_model=JobStartResponse)
    async def create_application_job(
        resume_text: str = Form(""),
        job_description_text: str = Form(...),
        recruiter_questions: str = Form(""),
        enable_company_search: bool = Form(True),
        resume_file: UploadFile | None = File(None),
    ) -> JobStartResponse:
        resume_text = resume_text.strip()
        job_description_text = job_description_text.strip()
        resume_file_metadata: dict[str, Any] | None = None

        if resume_file and resume_file.filename:
            try:
                resume_text, resume_file_metadata = await extract_resume_text(resume_file)
            except DocumentExtractionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        if len(resume_text) < 20:
            raise HTTPException(status_code=422, detail="Upload a supported resume document.")

        if len(job_description_text) < 20:
            raise HTTPException(status_code=422, detail="Paste a job description before submitting.")

        questions = [line.strip() for line in recruiter_questions.splitlines() if line.strip()]
        job_id = create_job(resume_file_metadata)
        update_step(
            job_id,
            "extract_resume",
            "completed",
            f"Extracted {resume_file_metadata['filename']}."
            if resume_file_metadata
            else "Using resume text provided through the API.",
        )

        thread = Thread(
            target=run_application_job,
            args=(job_id, resume_text, job_description_text, questions, enable_company_search, resume_file_metadata),
            daemon=True,
        )
        thread.start()

        return JobStartResponse(job_id=job_id)

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    def get_application_job(job_id: str) -> JobStatusResponse:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found.")
            return JobStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                steps=job["steps"],
                result=job["result"],
                error=job["error"],
                resume_file=job["resume_file"],
                chat_history=job.get("chat_history", []),
                latest_cover_letter=job.get("latest_cover_letter"),
            )

    @app.post("/api/jobs/{job_id}/chat", response_model=ChatResponse)
    async def chat_with_application(job_id: str, payload: ChatRequest) -> ChatResponse:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found.")
            if job["status"] != "completed" or not job["result"]:
                raise HTTPException(status_code=409, detail="Application job is not complete yet.")

            application_context = deepcopy(job["result"])
            chat_history = deepcopy(job.get("chat_history", []))
            latest_cover_letter = payload.cover_letter_text or job.get("latest_cover_letter")

        next_history = [
            *chat_history,
            {"role": "user", "content": payload.message, "created_at": utc_now()},
        ]

        try:
            system = get_application_system()
            chat_result = await run_in_threadpool(
                system.chat_about_application,
                application_context,
                [{"role": item["role"], "content": item["content"]} for item in next_history],
                latest_cover_letter,
                payload.enable_web_search,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ollama chat failed: {exc}") from exc

        revised_cover_letter = chat_result.get("cover_letter_text")
        cover_letter_updated = bool(revised_cover_letter)
        current_cover_letter = revised_cover_letter or latest_cover_letter
        assistant_message = {
            "role": "assistant",
            "content": chat_result["reply"],
            "created_at": utc_now(),
            "cover_letter_updated": cover_letter_updated,
            "suggestions": chat_result.get("suggestions", []),
        }
        if chat_result.get("web_search"):
            assistant_message["web_search"] = chat_result["web_search"]
        if cover_letter_updated:
            assistant_message["cover_letter_text"] = revised_cover_letter

        updated_history = [*next_history, assistant_message][-30:]

        with JOBS_LOCK:
            job = JOBS[job_id]
            job["chat_history"] = updated_history
            job["latest_cover_letter"] = current_cover_letter
            job["updated_at"] = utc_now()

        return ChatResponse(
            reply=chat_result["reply"],
            cover_letter_text=current_cover_letter,
            cover_letter_updated=cover_letter_updated,
            suggestions=chat_result.get("suggestions", []),
            messages=updated_history,
            web_search=chat_result.get("web_search"),
        )

    if FRONTEND_ASSETS_DIR.exists():
        app.mount("/_next", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="next-assets")

    if FRONTEND_OUT_DIR.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=FRONTEND_OUT_DIR, check_dir=False),
            name="frontend-assets",
        )

        @app.get("/{path:path}", include_in_schema=False)
        def serve_frontend(path: str) -> FileResponse:
            requested = (FRONTEND_OUT_DIR / path).resolve()
            if path and requested.is_file() and FRONTEND_OUT_DIR in requested.parents:
                return FileResponse(requested)
            route_html = (FRONTEND_OUT_DIR / f"{path}.html").resolve()
            if path and route_html.is_file() and FRONTEND_OUT_DIR in route_html.parents:
                return FileResponse(route_html)
            route_index = (FRONTEND_OUT_DIR / path / "index.html").resolve()
            if path and route_index.is_file() and FRONTEND_OUT_DIR in route_index.parents:
                return FileResponse(route_index)
            return FileResponse(FRONTEND_OUT_DIR / "index.html")

    return app


app = build_app()
