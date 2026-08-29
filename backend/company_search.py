from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel, HttpUrl


class CompanySearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str


class CompanyResearch(BaseModel):
    query: str
    results: list[CompanySearchResult]
    error: str | None = None

    def as_context(self) -> str:
        if not self.results:
            return self.error or "No online company details found."

        lines = []
        for index, result in enumerate(self.results, start=1):
            lines.append(
                f"{index}. {result.title}\n"
                f"   URL: {result.url}\n"
                f"   Snippet: {result.snippet}"
            )
        return "\n".join(lines)


class CompanySearcher:
    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def search_query(self, query: str, limit: int = 5) -> CompanyResearch:
        query = query.strip()
        if not query:
            return CompanyResearch(query="", results=[], error="No search query was generated.")

        return self._search(query=query, limit=limit, empty_message="No web search results found.")

    def search(self, company_name: str, role: str | None = None, limit: int = 5) -> CompanyResearch:
        query = " ".join(
            part
            for part in [
                company_name,
                role or "",
                "company mission product careers about",
            ]
            if part.strip()
        )

        return self._search(query=query, limit=limit, empty_message="No company search results found.")

    def _search(self, query: str, limit: int, empty_message: str) -> CompanyResearch:
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 JobCopilot/0.1 (+https://localhost)",
                },
            ) as client:
                response = client.get(f"https://duckduckgo.com/html/?q={quote_plus(query)}")
                response.raise_for_status()
        except Exception as exc:
            return CompanyResearch(query=query, results=[], error=f"Web search failed: {exc}")

        soup = BeautifulSoup(response.text, "html.parser")
        results: list[CompanySearchResult] = []

        for result in soup.select(".result"):
            title_link = result.select_one(".result__a")
            if not title_link:
                continue

            href = title_link.get("href")
            href = self._normalize_result_url(href)
            title = title_link.get_text(" ", strip=True)
            snippet_node = result.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

            if not href or not title:
                continue

            if self._is_likely_personal_github_pages_url(href):
                continue

            try:
                results.append(
                    CompanySearchResult(
                        title=title,
                        url=href,
                        snippet=snippet,
                    )
                )
            except ValueError:
                continue

            if len(results) >= limit:
                break

        if not results:
            return CompanyResearch(query=query, results=[], error=empty_message)

        return CompanyResearch(query=query, results=results)

    @staticmethod
    def _normalize_result_url(href: str | None) -> str | None:
        if not href:
            return None

        if href.startswith("//"):
            href = f"https:{href}"

        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])

        return href

    @staticmethod
    def _is_likely_personal_github_pages_url(url: str) -> bool:
        """Treat GitHub Pages subdomains as unverified personal/project sites."""
        hostname = (urlparse(url).hostname or "").lower().rstrip(".")
        return hostname.endswith(".github.io")
