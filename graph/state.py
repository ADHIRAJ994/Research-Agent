from typing import TypedDict


class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    findings: list[dict]
    draft: str
    citations: list[dict]
    critique: str | None
    revision_count: int
    approved: bool