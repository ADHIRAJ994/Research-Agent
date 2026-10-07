from typing import TypedDict
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph,START,END
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
class ResearchState(TypedDict):
    question:str
    sub_questions:list[str]
    findings:list[dict]
    draft:str
    citations:list[dict]
    critique:str|None
    revision_count:int
    approved:bool


def passthrough_node(state:ResearchState)->ResearchState:
    print(f"Recieved question: {state['question']}")
    return state

builder = StateGraph(ResearchState)
builder.add_node("passthrough",passthrough_node)
builder.add_edge(START,"passthrough")
builder.add_edge("passthrough",END)

graph = builder.compile()
print("DONE COMPILING")
initial_state = {
    "question": "What are the current pricing strategies of AI note-taking apps?",
    "sub_questions": [],
    "findings": [],
    "draft": "",
    "citations": [],
    "critique": None,
    "revision_count": 0,
    "approved": False,
}

result = graph.invoke(initial_state)
print(result)