from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MAX_REVISIONS
from graph.state import ResearchState
from graph.nodes.writer import build_source_registry, extract_citations


class CriticVerdict(BaseModel):
    approved: bool = Field(
        description="True only if every sub-question is addressed, every factual claim "
                    "has a citation, and each cited source actually supports its claim."
    )
    critique: str = Field(
        description="If not approved: specific, actionable feedback naming the exact "
                    "sub-question or quoting the exact sentence that has a problem, and "
                    "saying what to fix. If approved: an empty string."
    )


def make_critic_llm():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, api_key=GROQ_API_KEY)
    return llm.with_structured_output(CriticVerdict)

def critique_draft(
    question: str,
    sub_questions: list[str],
    draft: str,
    sources_text: str,
    code_issues: list[str],
    critic_llm,
) -> CriticVerdict:
    """Asks the Critic LLM to review the draft against the sources."""

    numbered_questions = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1))
    detected = "\n".join(f"- {issue}" for issue in code_issues) if code_issues else "None"

    prompt = (
        "You are a strict editor reviewing a research report.\n\n"
        f"Research question: {question}\n\n"
        f"Sub-questions the report must address:\n{numbered_questions}\n\n"
        f"SOURCES (the only material the report is allowed to use):\n{sources_text}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"Problems already detected automatically:\n{detected}\n\n"
        "Check:\n"
        "(a) Every sub-question is addressed.\n"
        "(b) Every non-trivial factual claim has an [S#] citation, and the cited source "
        "actually supports that claim.\n"
        "(c) The Gaps section honestly lists what the sources did not cover.\n\n"
        "Approve only if all three pass AND there are no automatically detected problems. "
        "If you reject, your critique must name the exact sub-question or quote the exact "
        "sentence at fault and say what to fix. Do not write vague feedback like "
        "'improve clarity'."
    )

    return critic_llm.invoke(prompt)

def critic_node(state: ResearchState) -> ResearchState:
    """LangGraph node. Reads draft + findings, writes approved + critique.
    Force-approves once the revision cap is reached."""

    registry, sources_text = build_source_registry(state["findings"])
    citations, invalid_ids = extract_citations(state["draft"], registry)

    code_issues = []
    if invalid_ids:
        code_issues.append(
            f"The draft cites source IDs that do not exist: {sorted(set(invalid_ids))}. "
            "Remove them or replace them with real IDs from the source list."
        )
    if not citations:
        code_issues.append(
            "The draft contains no valid [S#] citations. Every factual claim needs one."
        )

    critic_llm = make_critic_llm()
    verdict = critique_draft(
        question=state["question"],
        sub_questions=state["sub_questions"],
        draft=state["draft"],
        sources_text=sources_text,
        code_issues=code_issues,
        critic_llm=critic_llm,
    )

    approved = verdict.approved and not code_issues

    parts = list(code_issues)
    if verdict.critique.strip():
        parts.append(verdict.critique.strip())
    critique_text = "\n".join(parts)
    if not approved and not critique_text:
        critique_text = ("Draft was not approved. Re-check that every sub-question is "
                         "answered and every claim carries a valid citation.")

    force_approved = False
    if not approved and state["revision_count"] >= MAX_REVISIONS:
        approved = True
        force_approved = True
        critique_text = (f"[FORCE-APPROVED after {state['revision_count']} revisions] "
                         f"Unresolved issues:\n{critique_text}")

    print(f"\n[Critic] approved={approved} force_approved={force_approved} "
          f"revision_count={state['revision_count']}/{MAX_REVISIONS}")
    if critique_text:
        print(f"[Critic] feedback:\n{critique_text}")

    return {
        **state,
        "approved": approved,
        "critique": None if (approved and not force_approved) else critique_text,
    }

if __name__ == "__main__":
    test_state = {
        "question": "What are the pricing strategies of AI note-taking apps?",
        "sub_questions": ["What does Notion AI cost?", "What does Mem cost?"],
        "findings": [
            {
                "sub_question": "What does Notion AI cost?",
                "summary": "Notion AI is an add-on.",
                "sources": [
                    {"title": "Notion pricing", "url": "https://www.notion.com/pricing",
                     "snippet": "Notion AI is available as a paid add-on, billed per member."},
                ],
            },
            {
                "sub_question": "What does Mem cost?",
                "summary": "Mem has a subscription.",
                "sources": [
                    {"title": "Mem pricing", "url": "https://get.mem.ai/pricing",
                     "snippet": "Mem offers a monthly subscription for its AI features."},
                ],
            },
        ],
        "draft": (
            "## Notion AI\n"
            "Notion AI is sold as a paid add-on [S1]. It is also the most popular option.\n\n"
            "## Gaps\n"
            "Pricing for other tools is covered in [S99]."
        ),
        "citations": [],
        "critique": None,
        "revision_count": 0,
        "approved": False,
    }

    result = critic_node(test_state)
    print("\napproved:", result["approved"])