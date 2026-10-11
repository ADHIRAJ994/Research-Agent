from langgraph.graph import StateGraph, START, END
from config import MAX_REVISIONS
from graph.state import ResearchState
from graph.nodes.planner import planner_node
from graph.nodes.researcher import researcher_node
from graph.nodes.writer import writer_node
from graph.nodes.critic import critic_node


def route_after_critic(state: ResearchState) -> str:
    """Decides where to go after the Critic: finish, or send back to the Writer."""
    return "end" if state["approved"] else "writer"


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"end": END, "writer": "writer"},
    )

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

    final_state = initial_state
    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"\n--- finished node: {node_name} ---")
            if node_name == "writer":
                print(f"revision_count={update['revision_count']} | "
                      f"words={len(update['draft'].split())}")
            if node_name == "critic":
                print(f"approved={update['approved']}")
            final_state = update

    print("\n=== FINAL REPORT ===\n")
    print(final_state["draft"])
    print(f"\nRevisions used: {final_state['revision_count']} / {MAX_REVISIONS}")