from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from graph.nodes.planner import planner_node
from graph.nodes.researcher import researcher_node
from graph.nodes.writer import writer_node


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
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

    print("\n=== FINAL DRAFT ===\n")
    print(result["draft"])
    print(f"\nCitations: {len(result['citations'])}")