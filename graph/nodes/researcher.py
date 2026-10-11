from graph.state import ResearchState
import os
from tools.web_search import make_web_search_tool
from tools.faiss_retriever import make_faiss_retrieve_tool,build_or_load_faiss_index
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
def research_sub_question(sub_question: str, web_search, faiss_retrieve) -> dict:
    """Runs both tools for one sub-question, logs every call, and returns a
    findings entry with real, traceable sources."""

    print(f"\n[Researcher] Sub-question: {sub_question}")

    web_results = web_search(sub_question)
    print(f"  → web_search returned {len(web_results)} results")

    local_results = faiss_retrieve(sub_question)
    print(f"  → faiss_retrieve returned {len(local_results)} results")

    sources = []
    for r in web_results:
        sources.append({"tool": "web_search", "title": r["title"], "url": r["url"], "snippet": r["content"]})
    for r in local_results:
        sources.append({"tool": "faiss_retrieve", "title": r["title"], "url": None, "snippet": r["content"]})

    return {
        "sub_question": sub_question,
        "sources": sources,
    }

from langchain_groq import ChatGroq

def make_summarizer_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


def summarize_sources(sub_question: str, sources: list[dict], summarizer_llm: ChatGroq) -> str:
    """Condenses raw source snippets into a short summary answering the sub-question,
    while preserving which source each claim came from."""

    source_text = "\n\n".join(
        f"[Source {i+1}] {s['title']}\n{s['snippet']}"
        for i, s in enumerate(sources)
    )

    prompt = (
        f"Sub-question: {sub_question}\n\n"
        f"Sources:\n{source_text}\n\n"
        "Write a concise 2-4 sentence summary answering the sub-question, using only "
        "information present in the sources above. Reference sources by their number, "
        "e.g. 'according to Source 2'. Do not add outside knowledge."
    )

    response = summarizer_llm.invoke(prompt)
    return response.content

def researcher_node(state: ResearchState) -> ResearchState:
    """LangGraph node. Reads state['sub_questions'], writes state['findings']."""

    web_search = make_web_search_tool(TAVILY_API_KEY)
    index = build_or_load_faiss_index(source_dir="./data/knowledge_base")
    faiss_retrieve = make_faiss_retrieve_tool(index)
    summarizer_llm = make_summarizer_llm()

    findings = []
    for sub_question in state["sub_questions"]:
        result = research_sub_question(sub_question, web_search, faiss_retrieve)
        summary = summarize_sources(sub_question, result["sources"], summarizer_llm)

        findings.append({
            "sub_question": result["sub_question"],
            "sources": result["sources"],
            "summary": summary,
        })

    return {
        **state,
        "findings": findings,
    }