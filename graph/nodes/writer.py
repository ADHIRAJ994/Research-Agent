import re
from langchain_groq import ChatGroq

from config import GROQ_API_KEY
from graph.state import ResearchState

def normalize_citations(text: str) -> str:
    """Rewrites any citation bracket style into the canonical [S1][S4] form."""

    def fix(match):
        ids = re.findall(r"S\d+", match.group(1))
        return "".join(f"[{i}]" for i in ids)

    return re.sub(r"[\[【]\s*((?:S\d+\s*[,;]?\s*)+)[\]】]", fix, text)

def build_source_registry(findings: list[dict]) -> tuple[dict, str]:
    """Assigns every retrieved source a unique ID (S1, S2, ...) and builds the
    text block that gets shown to the Writer, grouped by sub-question."""

    registry = {}
    blocks = []
    counter = 1

    for finding in findings:
        blocks.append(f"### Sub-question: {finding['sub_question']}")
        blocks.append(f"Research summary: {finding['summary']}")
        blocks.append("Sources:")

        for source in finding["sources"]:
            source_id = f"S{counter}"
            registry[source_id] = {
                "title": source["title"],
                "url": source["url"],
                "snippet": source["snippet"],
                "sub_question": finding["sub_question"],
            }
            blocks.append(f"[{source_id}] {source['title']}: {source['snippet'][:400]}")
            counter += 1

        blocks.append("")

    return registry, "\n".join(blocks)

def make_writer_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, api_key=GROQ_API_KEY)


def write_draft(
    question: str,
    sources_text: str,
    writer_llm: ChatGroq,
    critique: str | None = None,
    previous_draft: str = "",
) -> str:
    """Writes the report. If a critique is given, revises previous_draft to fix it."""

    rules = (
        "Rules:\n"
        "- Use ONLY the sources provided below. Do not add outside knowledge.\n"
        "- After every factual claim, cite its source with its ID in square brackets, "
        "e.g. 'Notion AI costs $10/month [S3].'\n"
        "- Put each ID in its own brackets: write [S1][S4], never [S1, S4].\n"
        "- Only use IDs that appear in the source list. Never invent an ID.\n"
        "- Structure the report with a short intro, one section per sub-question "
        "(markdown ## headers), and a final 'Gaps' section listing anything the "
        "sources did not cover.\n"
    )

    if critique:
        prompt = (
            f"You are revising a research report.\n\n"
            f"Research question: {question}\n\n"
            f"{rules}\n"
            f"CURRENT DRAFT:\n{previous_draft}\n\n"
            f"EDITOR FEEDBACK (address every point):\n{critique}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            "Rewrite the complete report, fixing every issue the editor raised."
        )
    else:
        prompt = (
            f"You are a research analyst writing a report.\n\n"
            f"Research question: {question}\n\n"
            f"{rules}\n"
            f"SOURCES:\n{sources_text}\n\n"
            "Write the complete report now."
        )

    response = writer_llm.invoke(prompt)
    return normalize_citations(response.content)

def extract_citations(draft: str, registry: dict) -> tuple[list[dict], list[str]]:
    """Finds every [S#] marker in the draft and maps it to the real source.
    Returns (citations, invalid_ids). Invalid IDs are markers the model made up."""

    citations = []
    invalid_ids = []

    sentences = re.split(r"(?<=[.!?])\s+", draft)

    for sentence in sentences:
        for source_id in re.findall(r"\[(S\d+)\]", sentence):
            if source_id in registry:
                source = registry[source_id]
                citations.append({
                    "claim": sentence.strip(),
                    "source_id": source_id,
                    "source_title": source["title"],
                    "source_url": source["url"],
                })
            else:
                invalid_ids.append(source_id)

    return citations, invalid_ids

def writer_node(state: ResearchState) -> ResearchState:
    """LangGraph node. Reads findings (+ critique on revisions), writes draft and citations."""

    registry, sources_text = build_source_registry(state["findings"])
    writer_llm = make_writer_llm()

    is_revision = state["critique"] is not None

    draft = write_draft(
        question=state["question"],
        sources_text=sources_text,
        writer_llm=writer_llm,
        critique=state["critique"],
        previous_draft=state["draft"],
    )

    citations, invalid_ids = extract_citations(draft, registry)

    print(f"\n[Writer] Draft written ({len(draft.split())} words, "
          f"{len(citations)} citations, revision={is_revision})")
    if invalid_ids:
        print(f"[Writer] WARNING: draft cites IDs that don't exist: {sorted(set(invalid_ids))}")

    return {
        **state,
        "draft": draft,
        "citations": citations,
        "revision_count": state["revision_count"] + (1 if is_revision else 0),
    }

if __name__ == "__main__":
    test_state = {
        "question": "What are the pricing strategies of AI note-taking apps?",
        "sub_questions": ["What does Notion AI cost?"],
        "findings": [
            {
                "sub_question": "What does Notion AI cost?",
                "summary": "Notion AI is sold as an add-on to Notion plans.",
                "sources": [
                    {
                        "title": "Notion pricing",
                        "url": "https://www.notion.com/pricing",
                        "snippet": "Notion AI is available as an add-on for paid plans, billed per member per month.",
                    },
                    {
                        "title": "notes.txt",
                        "url": None,
                        "snippet": "Per-seat add-on pricing is common among note-taking tools.",
                    },
                ],
            }
        ],
        "draft": "",
        "citations": [],
        "critique": None,
        "revision_count": 0,
        "approved": False,
    }

    result = writer_node(test_state)
    print("\n=== DRAFT ===\n")
    print(result["draft"])
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(f"- [{c['source_id']}] {c['source_title']} -> {c['source_url']}")