
from pydantic import BaseModel, Field
from graph.state import ResearchState
from langchain_groq import ChatGroq
from config import GROQ_API_KEY


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description=(
            "2 to 5 specific, independently-researchable sub-questions "
            "that together fully address the main research question. "
            "No overlap between them. Ordered from foundational to specific."
        )
    )


def make_planner_llm():
    """
    Returns a Groq client configured for structured SubQuestions output.
    """

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing!")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY.strip(),
        temperature=0.4,
        max_retries=2
    )

    return llm.with_structured_output(
        SubQuestions,
        method="json_schema",
        strict=True
    )


def plan(question: str, planner_llm) -> list[str]:
    """
    Decompose a research question into 2-5 sub-questions.
    """

    prompt = (
        f"Research question: {question}\n\n"
        "Break this down into 2 to 5 specific, "
        "independently-researchable sub-questions.\n"
        "Do not answer the questions yourself.\n"
        "Return only the sub-questions according to the schema."
    )

    result: SubQuestions = planner_llm.invoke(prompt)

    return result.sub_questions


planner_llm = make_planner_llm()


def planner_node(state: ResearchState) -> ResearchState:
    """LangGraph node wrapper around plan()."""

    sub_questions = plan(
        state["question"],
        planner_llm
    )

    return {
        **state,
        "sub_questions": sub_questions
    }


# Standalone testing
if __name__ == "__main__":

    test_state = {
        "question": "What are the current pricing strategies of AI note-taking apps?",
        "sub_questions": [],
        "findings": [],
        "draft": "",
        "citations": [],
        "critique": None,
        "revision_count": 0,
        "approved": False,
    }

    print(planner_node(test_state))
