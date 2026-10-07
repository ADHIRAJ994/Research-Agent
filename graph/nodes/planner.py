from pydantic import BaseModel, Field
from dotenv import load_dotenv
from graph.state import ResearchState
from langchain_groq import ChatGroq

load_dotenv()

class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description=(
            "2 to 5 specific, independently-researchable sub-questions "
            "that together fully address the main research question. "
            "No overlap between them. Ordered from foundational to specific."
        )
    )

def make_planner_llm() -> ChatGroq:
    """
    Returns a Gemini client configured to return
    a SubQuestions object.
    """

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.4,
    )

    return llm.with_structured_output(SubQuestions,method="json_schema",strict=True)

def plan(question: str,planner_llm: ChatGroq) -> list[str]:
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

def planner_node(state:ResearchState)->ResearchState:
    """LangGraph node wrapper around plan(). Reads state['question'], writes state['sub_questions']."""
    sub_questions = plan(state["question"], planner_llm)
    return {
        **state,
        "sub_questions":sub_questions
    }
#Test

if __name__ == "__main__":

   test_state: ResearchState = {
        "question": "What are current pricing strategies of AI note-taking apps?",
        "sub_questions": []
    }

print("Input State:")
print(test_state)

result_state = planner_node(test_state)

print("OUTPUT STATES")
print(result_state)

print("\nGenerated Sub-Questions:")

for i, question in enumerate(
        result_state["sub_questions"],
        start=1
    ):
        print(f"{i}. {question}")