import json
from typing import Callable, Dict, List, Any, Optional, Iterable
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from datetime import datetime, date

from schema import (
    LangGraphApplicationState,
    ParsedResume,
    ParsedJobDescription,
    SkillMatchAnalysis,
    CoverLetter,
    RecruiterQuestion,
    ApplicationResult,
)

from company_search import CompanySearcher
from resume_rag import ResumeRAGConfig, build_resume_rag_context
from utils import extract_dict_from_json_response


class ResumeJobApplicationSystem:
    UNKNOWN_COMPANY_NAME = "Company Not Specified"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemma4:31b",
        host: Optional[str] = "http://localhost:11434",
        rag_config: Optional[ResumeRAGConfig] = None,
    ):
        """
        Initialize the system with Ollama integration via LangGraph

        Args:
            api_key: Optional Ollama API key for cloud or authenticated hosts
            model: Ollama model to use (default: gemma4:31b)
            host: Ollama host/base URL (default: http://localhost:11434)
        """
        self.rag_config = rag_config or ResumeRAGConfig.from_env()
        client_kwargs = {}
        if api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

        self.model = ChatOllama(
            model=model,
            base_url=host,
            client_kwargs=client_kwargs or None,
            num_predict=2048,
            temperature=0,
            timeout=200,
        )
        self.company_searcher = CompanySearcher()
        self.graph = self._build_graph()

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _coerce_string_fields(item: Dict[str, Any], fields: Iterable[str]) -> None:
        for field in fields:
            if item.get(field) is None:
                item[field] = ""

    @staticmethod
    def _coerce_list_fields(item: Dict[str, Any], fields: Iterable[str]) -> None:
        for field in fields:
            item[field] = ResumeJobApplicationSystem._ensure_list(item.get(field))

    @staticmethod
    def _truncate_text(value: Any, max_length: int = 3000) -> str:
        if value is None:
            return ""
        text = str(value)
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "\n...[truncated]"

    @staticmethod
    def _fallback_chat_suggestions() -> List[str]:
        return [
            "How can this cover letter sound more specific to the company?",
            "What skill gaps should I address before applying?",
            "How would you tighten the opening paragraph?",
        ]

    @staticmethod
    def _normalize_chat_suggestions(value: Any) -> List[str]:
        if not isinstance(value, list):
            return ResumeJobApplicationSystem._fallback_chat_suggestions()

        suggestions = []
        for item in value:
            if isinstance(item, str) and item.strip():
                suggestions.append(item.strip())
            if len(suggestions) == 3:
                break

        for fallback in ResumeJobApplicationSystem._fallback_chat_suggestions():
            if len(suggestions) == 3:
                break
            if fallback not in suggestions:
                suggestions.append(fallback)

        return suggestions[:3]

    @classmethod
    def _is_missing_company_name(cls, value: Any) -> bool:
        if not isinstance(value, str):
            return True
        return value.strip().lower() in {
            "",
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "not provided",
            "not specified",
            cls.UNKNOWN_COMPANY_NAME.lower(),
        }

    @classmethod
    def _company_reference(cls, value: Any) -> str:
        if cls._is_missing_company_name(value):
            return "the hiring team"
        return str(value).strip()

    @staticmethod
    def _infer_position_from_job_text(job_desc_text: str) -> str:
        for line in job_desc_text.splitlines():
            line = line.strip(" -\t")
            if line:
                return line[:120]
        return "Open Role"

    @staticmethod
    def _normalize_job_type(value: Any, job_desc_text: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        valid_types = {"full_time", "part_time", "contract", "freelance", "internship"}
        if normalized in valid_types:
            return normalized

        text = job_desc_text.lower()
        if "intern" in text:
            return "internship"
        if "contract" in text:
            return "contract"
        if "freelance" in text:
            return "freelance"
        if "part-time" in text or "part time" in text:
            return "part_time"
        return "full_time"

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "y", "1", "remote", "hybrid"}
        return False

    def _normalize_job_payload(self, parsed_job: Dict[str, Any], job_desc_text: str) -> Dict[str, Any]:
        if not isinstance(parsed_job, dict):
            parsed_job = {}

        company = parsed_job.get("company")
        if not isinstance(company, dict):
            company = {}
        if self._is_missing_company_name(company.get("name")):
            company["name"] = self.UNKNOWN_COMPANY_NAME
        self._coerce_string_fields(company, ["name"])
        for field in ["size", "location", "website", "description"]:
            if company.get(field) == "":
                company[field] = None
            else:
                company.setdefault(field, None)
        parsed_job["company"] = company

        if not isinstance(parsed_job.get("position"), str) or not parsed_job["position"].strip():
            parsed_job["position"] = self._infer_position_from_job_text(job_desc_text)

        if parsed_job.get("location") == "":
            parsed_job["location"] = None
        elif "location" not in parsed_job:
            parsed_job["location"] = None

        parsed_job["job_type"] = self._normalize_job_type(parsed_job.get("job_type"), job_desc_text)
        parsed_job["remote_options"] = self._normalize_bool(parsed_job.get("remote_options"))
        if parsed_job.get("salary_range") == "":
            parsed_job["salary_range"] = None
        parsed_job.setdefault("summary", "")
        parsed_job.setdefault("application_deadline", None)
        self._coerce_list_fields(
            parsed_job,
            ["responsibilities", "requirements", "preferred_qualifications", "benefits", "company_culture"],
        )

        normalized_requirements = []
        for requirement in parsed_job["requirements"]:
            if not isinstance(requirement, dict):
                continue
            if not isinstance(requirement.get("skill"), str) or not requirement["skill"].strip():
                continue
            if requirement.get("importance") not in {"required", "preferred", "nice_to_have"}:
                requirement["importance"] = "required"
            if requirement.get("category") not in {
                "technical",
                "soft",
                "domain_specific",
                "language",
                "certification",
            }:
                requirement["category"] = "technical"
            if not isinstance(requirement.get("years_required"), int):
                requirement["years_required"] = None
            if requirement.get("description") == "":
                requirement["description"] = None
            normalized_requirements.append(requirement)

        parsed_job["requirements"] = normalized_requirements

        return parsed_job

    def _normalize_resume_payload(self, parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
        personal_info = parsed_resume.get("personal_info")
        if isinstance(personal_info, dict):
            self._coerce_string_fields(personal_info, ["name"])

        for experience in self._ensure_list(parsed_resume.get("work_experiences")):
            if not isinstance(experience, dict):
                continue
            self._coerce_string_fields(experience, ["company", "position", "start_date", "end_date"])
            if experience.get("duration_months") is None:
                experience["duration_months"] = 0
            self._coerce_list_fields(
                experience,
                ["responsibilities", "achievements", "skills_used", "skills_learned", "technologies"],
            )

        for education in self._ensure_list(parsed_resume.get("education")):
            if not isinstance(education, dict):
                continue
            self._coerce_string_fields(education, ["institution", "degree", "field_of_study"])
            self._coerce_list_fields(education, ["relevant_coursework", "honors", "activities"])

        for project in self._ensure_list(parsed_resume.get("projects")):
            if not isinstance(project, dict):
                continue
            self._coerce_string_fields(project, ["name", "description"])
            self._coerce_list_fields(project, ["technologies", "achievements"])

        for certification in self._ensure_list(parsed_resume.get("certifications")):
            if not isinstance(certification, dict):
                continue
            self._coerce_string_fields(certification, ["name", "issuing_organization"])

        for language in self._ensure_list(parsed_resume.get("languages")):
            if not isinstance(language, dict):
                continue
            self._coerce_string_fields(language, ["name"])
            if language.get("proficiency") is None:
                language["proficiency"] = "basic"

        for skill in self._ensure_list(parsed_resume.get("skills")):
            if not isinstance(skill, dict):
                continue
            self._coerce_string_fields(skill, ["name", "context"])
            if skill.get("category") is None:
                skill["category"] = "technical"
            if skill.get("proficiency_level") is None:
                skill["proficiency_level"] = "intermediate"

        for field in ["work_experiences", "education", "skills", "certifications", "projects", "languages"]:
            parsed_resume[field] = self._ensure_list(parsed_resume.get(field))
        self._coerce_list_fields(parsed_resume, ["volunteer_experience", "publications", "awards"])

        return parsed_resume

    def _build_application_chat_context(
        self,
        application_context: Dict[str, Any],
        latest_cover_letter: Optional[str],
    ) -> Dict[str, Any]:
        parsed_resume = application_context.get("parsed_resume") or {}
        parsed_job = application_context.get("parsed_job_description") or {}
        skill_match = application_context.get("skill_match_analysis") or {}

        return {
            "summary": application_context.get("summary") or {},
            "candidate": {
                "personal_info": parsed_resume.get("personal_info"),
                "professional_summary": parsed_resume.get("professional_summary"),
                "work_experiences": (parsed_resume.get("work_experiences") or [])[:4],
                "education": parsed_resume.get("education"),
                "skills": (parsed_resume.get("skills") or [])[:25],
                "projects": (parsed_resume.get("projects") or [])[:5],
            },
            "job": parsed_job,
            "company_research": application_context.get("company_research"),
            "company_research_text": self._truncate_text(application_context.get("company_research_text"), 2500),
            "resume_rag_context": self._truncate_text(application_context.get("resume_rag_context"), 2500),
            "skill_match": {
                "overall_match_score": skill_match.get("overall_match_score"),
                "required_skills_match_score": skill_match.get("required_skills_match_score"),
                "skill_gaps": skill_match.get("skill_gaps"),
                "recommendations": skill_match.get("recommendations"),
            },
            "latest_cover_letter": self._truncate_text(
                latest_cover_letter or application_context.get("cover_letter_text"),
                4500,
            ),
        }

    @staticmethod
    def _latest_user_message(chat_history: List[Dict[str, Any]]) -> str:
        for item in reversed(chat_history):
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                return item["content"].strip()
        return ""

    def _fallback_chat_search_query(
        self,
        compact_context: Dict[str, Any],
        latest_user_message: str,
    ) -> str:
        job = compact_context.get("job") if isinstance(compact_context.get("job"), dict) else {}
        company = job.get("company") if isinstance(job.get("company"), dict) else {}
        company_name = company.get("name")
        position = job.get("position")

        parts = []
        if not self._is_missing_company_name(company_name):
            parts.append(str(company_name))
        if isinstance(position, str) and position.strip():
            parts.append(position.strip())
        if latest_user_message:
            parts.append(latest_user_message)

        return " ".join(parts).strip()[:240]

    def _build_chat_web_search_query(
        self,
        compact_context: Dict[str, Any],
        recent_history: List[Dict[str, Any]],
    ) -> str:
        latest_user_message = self._latest_user_message(recent_history)
        fallback_query = self._fallback_chat_search_query(compact_context, latest_user_message)
        if not latest_user_message:
            return fallback_query

        job = compact_context.get("job") if isinstance(compact_context.get("job"), dict) else {}
        company = job.get("company") if isinstance(job.get("company"), dict) else {}
        search_hint = {
            "current_date": date.today().isoformat(),
            "candidate_question": latest_user_message,
            "company": company.get("name"),
            "role": job.get("position"),
            "job_location": job.get("location"),
        }

        system_prompt = """
        You turn a job application chat question into one precise web search query.
        Include the company, role, product, location, or timeframe only when helpful.
        Return only valid JSON and do not answer the question.
        """

        prompt = f"""
        Build one search query for this chat request.

        CONTEXT:
        {json.dumps(search_hint, indent=2)}

        RECENT CHAT:
        {json.dumps(recent_history[-6:], indent=2)}

        Return exactly this JSON shape:
        {{
            "query": "short web search query"
        }}
        """

        try:
            response = self._call_ollama(prompt, system_prompt)
            parsed = extract_dict_from_json_response(response)
        except Exception:
            return fallback_query

        query = parsed.get("query")
        if not isinstance(query, str) or not query.strip():
            return fallback_query

        return " ".join(query.split())[:240]

    def chat_about_application(
        self,
        application_context: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
        latest_cover_letter: Optional[str] = None,
        enable_web_search: bool = False,
    ) -> Dict[str, Any]:
        """Answer follow-up questions or revise the cover letter using completed job context."""
        compact_context = self._build_application_chat_context(application_context, latest_cover_letter)
        recent_history = chat_history[-12:]
        web_search = None
        web_search_context = "No live web search was requested."

        if enable_web_search:
            search_query = self._build_chat_web_search_query(compact_context, recent_history)
            search_result = self.company_searcher.search_query(search_query, limit=5)
            web_search = search_result.model_dump(mode="json")
            web_search_context = self._truncate_text(
                f"Query: {search_result.query}\n{search_result.as_context()}",
                3500,
            )

        system_prompt = """
        You are JobCopilot's follow-up chat assistant. Use the provided application memory to answer
        candidate questions and revise the cover letter.

        Rules:
        - Use the resume, job description, company research, retrieved resume excerpts, skill match, and latest cover letter.
        - If online search results are provided, use them for current public facts and cite the source URLs from those results in the reply.
        - If online search was requested but no useful results were found, say that clearly and continue from application memory.
        - Never cite sources that are not present in the online search results.
        - If the user asks for a cover letter change, return a complete revised cover letter in cover_letter_text.
        - If the user only asks a question, set cover_letter_text to null.
        - Generate exactly three short suggested next questions based on the latest user question and your answer.
        - Keep answers specific to the stored application context. Do not invent candidate experience or company facts.
        - Return only valid JSON.
        """

        prompt = f"""
        APPLICATION MEMORY:
        {json.dumps(compact_context, indent=2)}

        CHAT HISTORY:
        {json.dumps(recent_history, indent=2)}

        ONLINE SEARCH MODE:
        {"enabled" if enable_web_search else "disabled"}

        ONLINE SEARCH RESULTS:
        {web_search_context}

        Return exactly this JSON shape:
        {{
            "reply": "concise assistant response",
            "cover_letter_text": "full revised cover letter when changed, otherwise null",
            "suggestions": [
                "context-aware suggested next question",
                "context-aware suggested next question",
                "context-aware suggested next question"
            ]
        }}
        """

        response = self._call_ollama(prompt, system_prompt)

        try:
            parsed = extract_dict_from_json_response(response)
        except ValueError:
            return {
                "reply": response.strip(),
                "cover_letter_text": None,
                "suggestions": self._fallback_chat_suggestions(),
                "web_search": web_search,
            }

        reply = parsed.get("reply")
        cover_letter_text = parsed.get("cover_letter_text")
        suggestions = self._normalize_chat_suggestions(parsed.get("suggestions"))

        if not isinstance(reply, str) or not reply.strip():
            reply = "I updated the application context, but the model did not return a readable reply."
        if not isinstance(cover_letter_text, str) or not cover_letter_text.strip():
            cover_letter_text = None

        return {
            "reply": reply.strip(),
            "cover_letter_text": cover_letter_text.strip() if cover_letter_text else None,
            "suggestions": suggestions,
            "web_search": web_search,
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(LangGraphApplicationState)

        workflow.add_node("parse_resume", self.parse_resume_node)
        workflow.add_node("parse_job_description", self.parse_job_description_node)
        workflow.add_node("research_company", self.research_company_node)
        workflow.add_node("retrieve_resume_context", self.retrieve_resume_context_node)
        workflow.add_node("analyze_skill_match", self.analyze_skill_match_node)
        workflow.add_node("generate_cover_letter", self.generate_cover_letter_node)
        workflow.add_node("handle_recruiter_questions", self.handle_recruiter_questions_node)
        workflow.add_node("finalize_application", self.finalize_application_node)

        workflow.set_entry_point("parse_resume")

        workflow.add_edge("parse_resume", "parse_job_description")
        workflow.add_edge("parse_job_description", "research_company")
        workflow.add_edge("research_company", "retrieve_resume_context")
        workflow.add_edge("retrieve_resume_context", "analyze_skill_match")
        workflow.add_edge("analyze_skill_match", "generate_cover_letter")
        workflow.add_edge("generate_cover_letter", "handle_recruiter_questions")
        workflow.add_edge("handle_recruiter_questions", "finalize_application")
        workflow.add_edge("finalize_application", END)

        return workflow.compile()

    @staticmethod
    def _emit_progress(
        state: LangGraphApplicationState,
        step: str,
        status: str,
        message: str,
    ) -> None:
        callback = state.get("progress_callback")
        if callable(callback):
            callback(step, status, message)

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to Ollama API via LangChain"""
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Ollama API error: {e}")
            raise

    def parse_resume_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Parse resume text into structured Pydantic model using Ollama"""
        self._emit_progress(state, "parse_resume", "running", "Parsing resume text and extracting structured profile data.")
        try:
            resume_text = state["resume_text"]
            core = self._parse_resume_content_with_ollama(resume_text)
            skills = self._parse_resume_skills_with_ollama(resume_text)
            parsed_resume = self._normalize_resume_payload({**core, "skills": skills})

            validated_resume = ParsedResume.model_validate(parsed_resume)

            state["parsed_resume"] = validated_resume
            state["current_step"] = "resume_parsed"
            self._emit_progress(state, "parse_resume", "completed", "Resume parsed and validated.")

            state["messages"] = [AIMessage(content="✅ Resume parsing completed with Ollama + Pydantic validation")]
            print("✅ Resume parsing completed with Ollama + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Resume parsing error: {str(e)}")
            self._emit_progress(state, "parse_resume", "failed", f"Resume parsing failed: {e}")
            state["messages"] = [AIMessage(content=f"❌ Resume parsing failed: {e}")]
            print(f"❌ Resume parsing failed: {e}")

        return state

    def parse_job_description_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Parse job description into structured Pydantic model using Ollama"""
        self._emit_progress(state, "parse_job", "running", "Parsing the job description and role requirements.")
        try:
            job_desc_text = state["job_description_text"]
            parsed_job = self._parse_job_description_content_with_ollama(job_desc_text)
            parsed_job = self._normalize_job_payload(parsed_job, job_desc_text)

            validated_job = ParsedJobDescription.model_validate(parsed_job)

            state["parsed_job_description"] = validated_job
            state["current_step"] = "job_description_parsed"
            self._emit_progress(state, "parse_job", "completed", "Job description parsed and validated.")

            # Update messages
            new_message = AIMessage(content="✅ Job description parsing completed with Ollama + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Job description parsing completed with Ollama + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Job description parsing error: {str(e)}")
            self._emit_progress(state, "parse_job", "failed", f"Job parsing failed: {e}")
            error_message = AIMessage(content=f"❌ Job description parsing failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Job description parsing failed: {e}")

        return state

    def research_company_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Search online for company details when enabled."""
        if not state.get("enable_company_search"):
            state["company_research"] = None
            state["company_research_text"] = None
            self._emit_progress(state, "company_search", "skipped", "Online company search is disabled.")
            return state

        self._emit_progress(state, "company_search", "running", "Searching online for company details.")

        try:
            job = state.get("parsed_job_description")
            if not job:
                raise ValueError("Missing parsed job description")
            if self._is_missing_company_name(job.company.name):
                message = "Company name was not provided in the job description; online company search skipped."
                state["company_research"] = {"query": "", "results": [], "error": message}
                state["company_research_text"] = message
                self._emit_progress(state, "company_search", "skipped", message)
                return state

            research = self.company_searcher.search(job.company.name, job.position)
            state["company_research"] = research.model_dump(mode="json")
            state["company_research_text"] = research.as_context()

            if research.error and not research.results:
                self._emit_progress(state, "company_search", "completed", research.error)
            else:
                self._emit_progress(
                    state,
                    "company_search",
                    "completed",
                    f"Found {len(research.results)} company detail sources.",
                )
        except Exception as e:
            state["company_research"] = {"query": "", "results": [], "error": str(e)}
            state["company_research_text"] = f"Company research unavailable: {e}"
            self._emit_progress(state, "company_search", "failed", f"Company search failed: {e}")

        return state

    def retrieve_resume_context_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Retrieve relevant resume excerpts for the job and recruiter questions."""
        self._emit_progress(state, "resume_rag", "running", "Retrieving resume excerpts relevant to the role.")

        try:
            resume_text = state["resume_text"]
            resume_metadata = state.get("resume_metadata") or {}
            job = state.get("parsed_job_description")
            if not job:
                raise ValueError("Missing parsed job description")

            requirement_query = "\n".join(
                [
                    job.position,
                    self._company_reference(job.company.name),
                    *job.responsibilities,
                    *[requirement.skill for requirement in job.requirements],
                    *(state.get("recruiter_questions") or []),
                ]
            )
            filename = resume_metadata.get("filename") if isinstance(resume_metadata, dict) else None
            state["resume_rag_context"] = build_resume_rag_context(
                resume_text,
                [state["job_description_text"], requirement_query],
                top_k=self.rag_config.top_k,
                config=self.rag_config,
                metadata={
                    "source": filename or "resume",
                    "doc_id": filename or "resume",
                },
            )
            self._emit_progress(
                state,
                "resume_rag",
                "completed",
                f"Relevant resume excerpts retrieved with {self.rag_config.retrieval_mode} mode.",
            )
        except Exception as e:
            state["resume_rag_context"] = state.get("resume_text", "")[:4000]
            self._emit_progress(state, "resume_rag", "failed", f"RAG fallback used: {e}")

        return state

    def analyze_skill_match_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Analyze skill matching with Ollama and Pydantic models"""
        self._emit_progress(state, "skill_match", "running", "Analyzing resume fit against job requirements.")
        try:
            resume = state["parsed_resume"]
            job = state["parsed_job_description"]

            if not resume or not job:
                raise ValueError("Missing parsed resume or job description")

            skill_analysis = self._analyze_skill_matching_with_ollama(
                resume,
                job,
                resume_rag_context=state.get("resume_rag_context"),
            )

            validated_analysis = SkillMatchAnalysis.model_validate(skill_analysis)

            state["skill_match_analysis"] = validated_analysis
            state["current_step"] = "skill_analysis_completed"
            self._emit_progress(state, "skill_match", "completed", "Skill match analysis completed.")

            # Update messages
            new_message = AIMessage(content="✅ Skill matching analysis completed with Ollama + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Skill matching analysis completed with Ollama + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Skill analysis error: {str(e)}")
            self._emit_progress(state, "skill_match", "failed", f"Skill analysis failed: {e}")
            error_message = AIMessage(content=f"❌ Skill analysis failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Skill analysis failed: {e}")

        return state

    def generate_cover_letter_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Generate tailored cover letter with Ollama and Pydantic model"""
        self._emit_progress(state, "cover_letter", "running", "Drafting the tailored cover letter.")
        try:
            resume = state["parsed_resume"]
            job = state["parsed_job_description"]
            skill_analysis = state["skill_match_analysis"]

            if not all([resume, job, skill_analysis]):
                raise ValueError("Missing required data for cover letter generation")

            cover_letter = self._generate_cover_letter_with_ollama(
                resume,
                job,
                skill_analysis,
                resume_rag_context=state.get("resume_rag_context"),
                company_research_text=state.get("company_research_text"),
            )

            validated_cover_letter = CoverLetter.model_validate(cover_letter)

            state["cover_letter"] = validated_cover_letter
            state["current_step"] = "cover_letter_generated"
            self._emit_progress(state, "cover_letter", "completed", "Cover letter generated.")

            # Update messages
            new_message = AIMessage(content="✅ Cover letter generated with Ollama + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Cover letter generated with Ollama + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Cover letter generation error: {str(e)}")
            self._emit_progress(state, "cover_letter", "failed", f"Cover letter generation failed: {e}")
            error_message = AIMessage(content=f"❌ Cover letter generation failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Cover letter generation failed: {e}")

        return state

    def handle_recruiter_questions_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Handle recruiter questions with Ollama and Pydantic models"""
        self._emit_progress(state, "recruiter_questions", "running", "Preparing recruiter question answers.")
        try:
            if state.get("recruiter_questions"):
                resume = state["parsed_resume"]
                job = state["parsed_job_description"]
                questions = state["recruiter_questions"]

                if not resume or not job:
                    raise ValueError("Missing parsed resume or job description")

                answers = self._answer_recruiter_questions_with_ollama(
                    resume,
                    job,
                    questions,
                    resume_rag_context=state.get("resume_rag_context"),
                    company_research_text=state.get("company_research_text"),
                )

                # Validate the answers
                validated_answers = [RecruiterQuestion.model_validate(answer) for answer in answers]

                state["recruiter_answers"] = validated_answers
                new_message = AIMessage(content="✅ Recruiter questions answered with Ollama + Pydantic validation")
                print("✅ Recruiter questions answered with Ollama + Pydantic validation")
            else:
                new_message = AIMessage(content="ℹ️ Follow-up chat context prepared")
                print("ℹ️ Follow-up chat context prepared")

            state["current_step"] = "recruiter_questions_handled"
            state["messages"] = state.get("messages", []) + [new_message]
            self._emit_progress(state, "recruiter_questions", "completed", "Follow-up chat context is ready.")

        except Exception as e:
            state["errors"].append(f"Recruiter questions handling error: {str(e)}")
            self._emit_progress(state, "recruiter_questions", "failed", f"Recruiter question handling failed: {e}")
            error_message = AIMessage(content=f"❌ Recruiter questions handling failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Recruiter questions handling failed: {e}")

        return state

    def finalize_application_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Finalize the application process"""
        if state["errors"]:
            state["current_step"] = "application_failed"
            message = "Application processing stopped because one or more required steps failed."
            self._emit_progress(state, "finalize", "failed", message)
            state["messages"] = state.get("messages", []) + [AIMessage(content=f"❌ {message}")]
            print(f"❌ {message}")
            return state

        state["current_step"] = "application_completed"
        self._emit_progress(state, "finalize", "completed", "Application response is ready.")
        final_message = AIMessage(content="🎉 Application processing completed with Ollama + Pydantic validation!")
        state["messages"] = state.get("messages", []) + [final_message]
        print("🎉 Application processing completed with Ollama + Pydantic validation!")
        return state

    def _parse_resume_content_with_ollama(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume text using Ollama into structured format"""
        today = date.today().isoformat()
        system_prompt = f"""You are an expert resume parser. Extract and structure resume information into the specified JSON format. 
        Be thorough and accurate.
        Track skills learned vs skills used in each job.
        For work experience, if end date is Present, then end date is {today}.

        Extract and structure the following sections into JSON:
        - personal_info
        - professional_summary
        - work_experiences
        - education
        - certifications
        - projects
        - languages
        - volunteer_experience
        - publications
        - awards
        
        IMPORTANT: Return ONLY valid JSON. No explanations, no markdown formatting, no additional text."""

        prompt = f"""
        Parse the following resume and extract all information into a structured JSON format that matches this schema:

        {{
            "personal_info": {{
                "name": "string",
                "email": "string (email format or null)",
                "location": "string or null",
                "linkedin_url": "string (URL format or null)",
                "portfolio_url": "string (URL format or null)"
            }},
            "professional_summary": "string",
            "work_experiences": [
                {{
                    "company": "string",
                    "position": "string",
                    "start_date": "string",
                    "end_date": "string",
                    "duration_months": "integer",
                    "location": "string or null",
                    "employment_type": "full_time|part_time|contract|freelance|internship",
                    "responsibilities": ["string", ...],
                    "achievements": ["string", ...],
                    "skills_used": ["string", ...],
                    "skills_learned": ["string", ...],
                    "technologies": ["string", ...],
                }}
            ],
            "education": [
                {{
                    "institution": "string",
                    "degree": "string",
                    "field_of_study": "string",
                    "level": "high_school|associate|bachelor|master|doctorate|certificate",
                    "graduation_date": "string or null",
                    "gpa": "number or null; preserve the original numeric grade scale, e.g. 3.6 for GPA, 8.2 for CGPA, 75 for 75%",
                    "relevant_coursework": ["string", ...],
                    "honors": ["string", ...],
                    "activities": ["string", ...]
                }}
            ],
            "certifications": [
                {{
                    "name": "string",
                    "issuing_organization": "string",
                    "issue_date": "string or null",
                    "expiry_date": "string or null",
                    "credential_id": "string or null"
                }}
            ],
            "projects": [
                {{
                    "name": "string",
                    "description": "string",
                    "technologies": ["string", ...],
                    "role": "string or null",
                    "achievements": ["string", ...]
                }}
            ],
            "languages": [
                {{
                    "name": "string",
                    "proficiency": "native|fluent|conversational|basic"
                }}
            ],
            "volunteer_experience": ["string", ...],
            "publications": ["string", ...],
            "awards": ["string", ...]
        }}

        Resume text:
        {resume_text}

        Return only the JSON object, no additional text.
        """

        response = self._call_ollama(prompt, system_prompt)
        return extract_dict_from_json_response(response)

    def _parse_resume_skills_with_ollama(self, resume_text: str) -> List[Dict[str, Any]]:
        """
        Parse only the skills section(s) of a resume into structured JSON via Ollama.
        Identifies skills learned vs used, categorizes, infers proficiency and years.
        """
        system_prompt = """
            You are an expert skills extractor. From the resume text, identify relevant skill mentioned
            and output a JSON array of skill objects.
            For now only extract maximum of 15 skills based on context and relevancy.
            For skills, identify both technical and soft skills, and categorize them appropriately.
            Calculate years of experience and proficiency levels based on context.
            For each skill, include:
            - name
            - category (technical|soft|domain_specific|language|certification)
            - proficiency_level (beginner|intermediate|advanced|expert)
            - years_experience (integer or null)
            - context (brief phrase: e.g. “used in X project” or “learned at Y”)

            IMPORTANT: Return ONLY valid JSON array. No extra text.
            """

        prompt = f"""
            Extract all skills from this resume text and structure them as:

            [
            {{
                "name": "string",
                "category": "technical|soft|domain_specific|language|certification",
                "proficiency_level": "beginner|intermediate|advanced|expert",
                "years_experience": integer or null,
                "context": "string or null"
            }},
            ...
            ]

            Resume text:
            {resume_text}

            Return only the JSON array.
            """

        response = self._call_ollama(prompt, system_prompt)
        return extract_dict_from_json_response(response, "list")

    def _parse_job_description_content_with_ollama(self, job_desc_text: str) -> Dict[str, Any]:
        """Parse job description using Ollama into structured format"""

        system_prompt = """You are an expert job description analyzer. Extract and structure job posting information into the specified JSON format.
        Identify requirements vs preferences, categorize skills, and extract company culture information accurately.
        If the job description does not name a company, set company.name to "Company Not Specified".
        If location, salary, benefits, deadline, or website are absent, use null or an empty list as appropriate.

        IMPORTANT: Return ONLY valid JSON. No explanations, no markdown formatting, no additional text."""

        prompt = f"""
        Parse the following job description and extract all information into a structured JSON format:

        {{
            "company": {{
                "name": "string",
                "size": "string or null",
                "location": "string or null",
                "website": "string (URL format or null)",
                "description": "string or null"
            }},
            "position": "string",
            "location": "string",
            "job_type": "full_time|part_time|contract|freelance|internship",
            "salary_range": "string or null",
            "remote_options": "boolean",
            "summary": "string",
            "responsibilities": ["string", ...],
            "requirements": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null",
                    "description": "string or null"
                }}
            ],
            "preferred_qualifications": ["string", ...],
            "benefits": ["string", ...],
            "company_culture": ["string", ...],
            "application_deadline": "string or null"
        }}

        Job description:
        {job_desc_text}

        Return only the JSON object, no additional text.
        """

        response = self._call_ollama(prompt, system_prompt)

        return extract_dict_from_json_response(response)

    def _analyze_skill_matching_with_ollama(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        resume_rag_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze skill matching using Ollama"""

        system_prompt = """You are an expert HR analyst specializing in skill matching and candidate assessment.
        Analyze how well a candidate's skills match job requirements. Provide detailed scoring and recommendations."""

        resume_skills = [
            {
                "name": skill.name,
                "category": skill.category,
                "proficiency": skill.proficiency_level,
                "years": skill.years_experience,
            }
            for skill in resume.skills
        ]

        job_requirements = [
            {
                "skill": req.skill,
                "importance": req.importance,
                "category": req.category,
                "years_required": req.years_required,
            }
            for req in job.requirements
        ]
        company_name = self._company_reference(job.company.name)

        scoring_prompt = f"""
        Analyze the skill match between this candidate's resume and job requirements:

        CANDIDATE SKILLS:
        {json.dumps(resume_skills, indent=2)}

        CANDIDATE EXPERIENCE:
        Total Years: {resume.get_total_experience_years()}
        Recent Positions: {[f"{exp.position} at {exp.company}" for exp in resume.work_experiences[:3]]}

        RETRIEVED RESUME EXCERPTS:
        {resume_rag_context or "No retrieved resume excerpts available."}

        JOB REQUIREMENTS:
        {json.dumps(job_requirements, indent=2)}

        JOB DETAILS:
        Position: {job.position}
        Company: {company_name}

        Provide a detailed skill match analysis in this JSON format:

        {{
            "overall_match_score": "float (0-100)",
            "required_skills_match_score": "float (0-100)",
            "matched_requirements": [
                {{
                    "requirement": {{
                        "skill": "string",
                        "importance": "required|preferred|nice_to_have",
                        "category": "technical|soft|domain_specific|language|certification",
                        "years_required": "integer or null"
                    }},
                    "resume_skill": {{
                        "name": "string",
                        "category": "technical|soft|domain_specific|language|certification",
                        "proficiency_level": "beginner|intermediate|advanced|expert",
                        "years_experience": "integer or null"
                    }},
                    "match_strength": "float (0.0-1.0)",
                    "gap_description": "string or null"
                }}
            ],
            "unmatched_requirements": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null"
                }}
            ],
            "skill_gaps": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null"
                }}
            ],
            "transferable_skills": [
                {{
                    "name": "string",
                    "category": "technical|soft|domain_specific|language|certification",
                    "proficiency_level": "beginner|intermediate|advanced|expert",
                    "years_experience": "integer or null"
                }}
            ]
        }}

        Return only the JSON object, no additional text.
        """

        scoring_response = self._call_ollama(scoring_prompt, system_prompt)
        scoring_data = extract_dict_from_json_response(scoring_response)

        recommendation_prompt = f"""
        Based on the following skill match analysis:

        {json.dumps(scoring_data, indent=2)}

        Provide a list of recommendations (as an array of strings) to help the candidate improve their fit for the job.
        
        Return in this JSON format:
        {{
            "recommendations": ["string", ...]
        }}
        """

        recommendation_response = self._call_ollama(recommendation_prompt, system_prompt)
        recommendation_data = extract_dict_from_json_response(recommendation_response)

        return {
            **scoring_data,
            **recommendation_data,
        }

    def _generate_cover_letter_with_ollama(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        skill_analysis: SkillMatchAnalysis,
        resume_rag_context: Optional[str] = None,
        company_research_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a tailored cover letter using Ollama"""

        system_prompt = """
                You are a senior career strategist and award-winning copywriter for tech roles.
                Your mission: craft a cover letter that
                1. Hooks the reader by naming a product/mission insight.
                2. Delivers a TL;DR with top achievements.
                3. Weaves a short narrative showing impact (with metrics).
                4. Lists core skills tied to role requirements.
                5. Connects personal values to company culture.
                6. Closes with a confident call to action.
                Ensure it scans well for ATS but reads naturally.
                """

        top_skills = [
            match.requirement.skill
            for match in sorted(skill_analysis.matched_requirements, key=lambda x: x.match_strength, reverse=True)[:5]
        ]

        recent = resume.work_experiences[0] if resume.work_experiences else None

        today = date.today().isoformat()
        company_name = self._company_reference(job.company.name)
        company_context_guidance = (
            f"Use researched company details for {company_name} when they are available."
            if not self._is_missing_company_name(job.company.name)
            else "No company name was provided. Address the hiring team and focus on the role, domain, and responsibilities instead of inventing company-specific facts."
        )
        core_responsibility = (
            job.responsibilities[0]
            if job.responsibilities
            else "the role's core ML engineering responsibilities"
        )

        prompt = f"""
        Use the SYSTEM instructions above and output EXACTLY this JSON:

        RETRIEVED RESUME EXCERPTS TO GROUND THE LETTER:
        {resume_rag_context or "No retrieved resume excerpts available."}

        ONLINE COMPANY DETAILS TO USE WHEN RELEVANT:
        {company_research_text or "No online company details available."}

        COMPANY CONTEXT GUIDANCE:
        {company_context_guidance}

        {{
            "header": "{resume.personal_info.name} | {resume.personal_info.email} | {resume.personal_info.location}\\n{today}",
            "tldr": "• {resume.get_total_experience_years()} yrs experience • Top skills: {", ".join(top_skills)} • Recent: {recent.position if recent else "N/A"} at {recent.company if recent else "N/A"}",
            "opening": "2-3 sentences. Start with a specific insight about {company_name} or, if the company is not named, the AI infrastructure/MLOps domain. Mention the {job.position} role by name.",
            "story_paragraph": "3-4 sentences. Describe a past project where you delivered X (metric) that maps directly to a core responsibility: {core_responsibility}.",
            "skills_paragraph": "3-4 sentences. Call out your top 3-5 skills ({", ".join(top_skills)}) and how each will solve a key challenge for {company_name}. Include one numeric result per skill.",
            "culture_fit": "2-3 sentences. Explain why {company_name}'s mission, values, or technical domain resonates with your career goals.",
            "closing": "2 sentences. Express enthusiasm, request next steps, and thank the reader.",
            "signature": "Best regards,\n{resume.personal_info.name}"
        }}

        Length: ~350 words total. No extra keys or commentary, only the JSON above.
        """

        response = self._call_ollama(prompt, system_prompt)

        return extract_dict_from_json_response(response)

    def _answer_recruiter_questions_with_ollama(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        questions: List[str],
        resume_rag_context: Optional[str] = None,
        company_research_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate answers to recruiter questions using Ollama"""

        system_prompt = """You are an expert interview coach and career counselor. 
        Generate thoughtful, professional answers to recruiter questions based on the candidate's background and the specific job opportunity."""

        candidate_context = {
            "name": resume.personal_info.name,
            "total_experience": resume.get_total_experience_years(),
            "recent_role": f"{resume.work_experiences[0].position} at {resume.work_experiences[0].company}"
            if resume.work_experiences
            else "N/A",
            "key_skills": [skill.name for skill in resume.skills[:10]],
            "achievements": resume.work_experiences[0].achievements[:3] if resume.work_experiences else [],
            "education": f"{resume.education[0].degree} in {resume.education[0].field_of_study}"
            if resume.education
            else "N/A",
        }

        job_context = {
            "position": job.position,
            "company": job.company.name,
            "responsibilities": job.responsibilities[:3],
        }

        prompt = f"""
        Generate professional answers to recruiter questions based on this candidate profile:

        CANDIDATE PROFILE:
        {json.dumps(candidate_context, indent=2)}

        JOB OPPORTUNITY:
        {json.dumps(job_context, indent=2)}

        RETRIEVED RESUME EXCERPTS:
        {resume_rag_context or "No retrieved resume excerpts available."}

        ONLINE COMPANY DETAILS:
        {company_research_text or "No online company details available."}

        RECRUITER QUESTIONS:
        {json.dumps(questions, indent=2)}

        For each question, provide a structured response in this JSON format:

        [
            {{
                "question": "string (the original question)",
                "category": "experience|motivation|skills|salary|availability|general",
                "answer": "string (2-3 sentences, professional and specific)",
                "confidence": "float (0.0-1.0, how confident the answer is)"
            }}
        ]

        Guidelines for answers:
        - Be specific and reference actual experience/skills
        - Show genuine interest in the role/company
        - Be confident but not arrogant
        - Keep answers concise but informative
        - Use the STAR method for experience questions
        - Be honest about salary expectations and availability

        Return only the JSON array, no additional text.
        """

        response = self._call_ollama(prompt, system_prompt)

        return extract_dict_from_json_response(response, type="list")

    def run_application_process(
        self,
        resume_text: str,
        job_description_text: str,
        recruiter_questions: Optional[List[str]] = None,
        enable_company_search: bool = True,
        resume_metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> ApplicationResult:
        """Run the complete application process with Ollama and return Pydantic result"""

        start_time = datetime.now()

        initial_state = LangGraphApplicationState(
            resume_text=resume_text,
            resume_metadata=resume_metadata,
            job_description_text=job_description_text,
            enable_company_search=enable_company_search,
            company_research=None,
            company_research_text=None,
            resume_rag_context=None,
            parsed_resume=None,
            parsed_job_description=None,
            skill_match_analysis=None,
            cover_letter=None,
            recruiter_questions=recruiter_questions,
            recruiter_answers=None,
            current_step="starting",
            errors=[],
            progress_callback=progress_callback,
            messages=[],  # Initialize messages list
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Create and validate the result using Pydantic
        result = ApplicationResult(
            parsed_resume=final_state.get("parsed_resume"),
            parsed_job_description=final_state.get("parsed_job_description"),
            skill_match_analysis=final_state.get("skill_match_analysis"),
            cover_letter=final_state.get("cover_letter"),
            recruiter_answers=final_state.get("recruiter_answers"),
            company_research=final_state.get("company_research"),
            company_research_text=final_state.get("company_research_text"),
            resume_rag_context=final_state.get("resume_rag_context"),
            errors=final_state.get("errors", []),
            status=final_state.get("current_step", "completed"),
            processing_time_seconds=processing_time,
        )

        return result
