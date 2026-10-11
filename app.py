import html
import re
import streamlit as st

from config import MAX_REVISIONS
from graph.build_graph import build_graph
from graph.nodes.writer import build_source_registry

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp textarea, .stApp button {
    font-family: 'Inter', sans-serif;
}
footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}

.hero {
    background: linear-gradient(135deg, #312E81 0%, #4F46E5 60%, #6366F1 100%);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
}
.hero-title {color: #FFFFFF; font-size: 2rem; font-weight: 700; line-height: 1.2;}
.hero-sub {color: #E0E7FF; font-size: 1rem; margin-top: 0.5rem; max-width: 720px;}

.pipeline {display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; margin: 0.8rem 0 1.4rem 0;}
.link {width: 22px; height: 2px; background: #CBD5E1;}
.stage {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.95rem; border-radius: 999px;
    border: 1px solid #E2E8F0; background: #FFFFFF;
    color: #64748B; font-size: 0.85rem; font-weight: 500;
}
.stage .dot {width: 8px; height: 8px; border-radius: 50%; background: #CBD5E1;}
.stage.active {border-color: #4F46E5; color: #4F46E5; background: #EEF2FF;}
.stage.active .dot {background: #4F46E5; animation: pulse 1s infinite;}
.stage.done {border-color: #A7F3D0; color: #047857; background: #ECFDF5;}
.stage.done .dot {background: #10B981;}
@keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.25;}}

.step {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #4F46E5;
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.step.ok {border-left-color: #10B981;}
.step.warn {border-left-color: #F59E0B;}
.step.bad {border-left-color: #EF4444;}
.step-title {font-weight: 600; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.25rem;}
.step-body {color: #475569; font-size: 0.88rem; line-height: 1.55;}

.stat {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.2rem;}
.stat-label {color: #64748B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;}
.stat-value {color: #0F172A; font-size: 1.6rem; font-weight: 700; margin-top: 0.15rem;}

.pill {display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;}
.pill.ok {background: #D1FAE5; color: #065F46;}
.pill.warn {background: #FEF3C7; color: #92400E;}
.pill.bad {background: #FEE2E2; color: #991B1B;}

.tool {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #EEF2FF; color: #3730A3;
    padding: 0.1rem 0.45rem; border-radius: 6px; font-size: 0.8rem;
}

.stButton button {border-radius: 10px; font-weight: 600; padding: 0.55rem 1.5rem;}
.stTextArea textarea {border-radius: 12px;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(text) -> str:
    """Escapes text before it goes into HTML, so retrieved or model-written
    text can never inject markup into the page."""
    return html.escape(str(text))


def pill(label: str, kind: str) -> str:
    return f'<span class="pill {kind}">{esc(label)}</span>'


def step_card(title: str, body_html: str, kind: str = "") -> str:
    return (
        f'<div class="step {kind}">'
        f'<div class="step-title">{esc(title)}</div>'
        f'<div class="step-body">{body_html}</div>'
        f"</div>"
    )


def stat_card(label: str, value) -> str:
    return (
        f'<div class="stat"><div class="stat-label">{esc(label)}</div>'
        f'<div class="stat-value">{esc(value)}</div></div>'
    )


STAGES = ["Planner", "Researcher", "Writer", "Critic"]
ALL_STAGES = set(STAGES)


def pipeline_html(done: set, active: str | None) -> str:
    parts = []
    for name in STAGES:
        if name in done:
            state = "done"
        elif name == active:
            state = "active"
        else:
            state = "idle"
        parts.append(f'<div class="stage {state}"><span class="dot"></span>{name}</div>')
    return '<div class="pipeline">' + '<div class="link"></div>'.join(parts) + "</div>"


def link_citations(draft: str, registry: dict) -> str:
    """Turns [S3] markers into clickable links (web sources) or labelled tags
    (local files). Unknown IDs are flagged as unverified, never linked."""

    def replace(match):
        source_id = match.group(1)
        source = registry.get(source_id)
        if source is None:
            return f"**\\[{source_id}: unverified\\]**"
        if source["url"]:
            return f"[\\[{source_id}\\]]({source['url']})"
        return f"**\\[{source_id}: {source['title']}\\]**"

    return re.sub(r"\[(S\d+)\]", replace, draft)


def humanize(text: str, registry: dict) -> str:
    """Appends the source title after every S-ID in free text."""

    def replace(match):
        sid = match.group(0)
        source = registry.get(sid)
        if source is None:
            return sid
        title = source["title"]
        if len(title) > 50:
            title = title[:47] + "..."
        return f'{sid} ("{title}")'

    return re.sub(r"\bS\d+\b", replace, text)


def render_report(draft: str, registry: dict) -> str:
    """Numbers citations 1, 2, 3 in order of first appearance and appends a
    References list containing only the sources actually cited."""

    order = []

    def replace(match):
        sid = match.group(1)
        source = registry.get(sid)
        if source is None:
            return "**\\[unverified\\]**"
        if sid not in order:
            order.append(sid)
        n = order.index(sid) + 1
        if source["url"]:
            return f"[\\[{n}\\]]({source['url']})"
        return f"**\\[{n}\\]**"

    body = re.sub(r"\[(S\d+)\]", replace, draft)

    refs = ["", "---", "#### References"]
    for n, sid in enumerate(order, 1):
        source = registry[sid]
        if source["url"]:
            refs.append(f"{n}. [{source['title']}]({source['url']})")
        else:
            refs.append(f"{n}. {source['title']} (local file)")

    return body + "\n" + "\n".join(refs)


# ---------------------------------------------------------------------------
# Header, sidebar, input
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Multi-Agent Research Assistant</div>'
    '<div class="hero-sub">Plans your question, searches the web and a local knowledge base, '
    "writes a cited report, and reviews its own draft before answering.</div>"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "**1. Planner** splits the question into sub-questions.\n\n"
        "**2. Researcher** runs web search and a local FAISS index for each one.\n\n"
        "**3. Writer** drafts a report that cites only retrieved sources.\n\n"
        "**4. Critic** checks coverage and citations, and sends weak drafts back."
    )
    st.markdown("---")
    st.caption(f"Revision cap: {MAX_REVISIONS}")

question = st.text_area(
    "Research question",
    placeholder="e.g. What are the current pricing strategies of AI note-taking apps?",
    height=100,
)
run_clicked = st.button("Run research", type="primary", disabled=not question.strip())

pipeline_slot = st.empty()
pipeline_slot.markdown(
    pipeline_html(ALL_STAGES if st.session_state.get("result") else set(), None),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Running the graph with a live trace
# ---------------------------------------------------------------------------

def run_pipeline(question: str, pipeline_slot):
    """Runs the graph, updating the stage tracker and a card log as each node finishes."""

    graph = build_graph()
    initial_state = {
        "question": question,
        "sub_questions": [],
        "findings": [],
        "draft": "",
        "citations": [],
        "critique": None,
        "revision_count": 0,
        "approved": False,
    }

    trace = {"drafts": [], "critiques": []}
    final_state = initial_state
    registry = {}
    done, active = set(), "Planner"
    pipeline_slot.markdown(pipeline_html(done, active), unsafe_allow_html=True)

    with st.status("Running agents", expanded=True) as status:
        for step in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in step.items():
                final_state = update

                if node_name == "planner":
                    done, active = {"Planner"}, "Researcher"
                    items = "".join(
                        f"<div>{i}. {esc(q)}</div>"
                        for i, q in enumerate(update["sub_questions"], 1)
                    )
                    st.markdown(step_card("Planner: sub-questions", items), unsafe_allow_html=True)

                elif node_name == "researcher":
                    registry, _ = build_source_registry(update["findings"])
                    done, active = {"Planner", "Researcher"}, "Writer"
                    lines = ""
                    for f in update["findings"]:
                        web = [s for s in f["sources"] if s.get("tool") == "web_search"]
                        local = [s for s in f["sources"] if s.get("tool") == "faiss_retrieve"]
                        lines += (
                            f"<div style='margin-bottom:0.5rem'><b>{esc(f['sub_question'])}</b><br>"
                            f"<span class='tool'>web_search</span> {len(web)} results &nbsp; "
                            f"<span class='tool'>faiss_retrieve</span> {len(local)} results</div>"
                        )
                    st.markdown(step_card("Researcher: tool calls", lines), unsafe_allow_html=True)

                elif node_name == "writer":
                    done, active = {"Planner", "Researcher"}, "Critic"
                    trace["drafts"].append(update["draft"])
                    words = len(update["draft"].split())
                    n_cites = len(update["citations"])
                    st.markdown(
                        step_card(
                            f"Writer: draft {len(trace['drafts'])}",
                            f"{words} words, {n_cites} citations",
                        ),
                        unsafe_allow_html=True,
                    )

                elif node_name == "critic":
                    forced = update["approved"] and update["critique"] is not None
                    trace["critiques"].append(
                        {"approved": update["approved"], "forced": forced, "critique": update["critique"]}
                    )
                    feedback = esc(humanize(update["critique"] or "", registry)).replace("\n", "<br>")

                    if forced:
                        done, active = ALL_STAGES, None
                        st.markdown(
                            step_card("Critic: force-approved (revision cap reached)", feedback, "warn"),
                            unsafe_allow_html=True,
                        )
                    elif update["approved"]:
                        done, active = ALL_STAGES, None
                        st.markdown(
                            step_card("Critic: approved", "The draft passed every check.", "ok"),
                            unsafe_allow_html=True,
                        )
                    else:
                        done, active = {"Planner", "Researcher"}, "Writer"
                        st.markdown(
                            step_card("Critic: rejected, sending back to the Writer", feedback, "bad"),
                            unsafe_allow_html=True,
                        )

                pipeline_slot.markdown(pipeline_html(done, active), unsafe_allow_html=True)

        status.update(label="Complete", state="complete", expanded=False)

    return final_state, trace


# ---------------------------------------------------------------------------
# Run on click, then render results
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        st.session_state["result"] = run_pipeline(question.strip(), pipeline_slot)
    except Exception as e:
        st.session_state["result"] = None
        pipeline_slot.markdown(pipeline_html(set(), None), unsafe_allow_html=True)
        st.error(f"The run failed: {e}")
        st.info("If this is a rate-limit (429) error from Groq, wait a minute and try again.")

result = st.session_state.get("result")

if result:
    final_state, trace = result
    registry, _ = build_source_registry(final_state["findings"])
    forced = final_state["approved"] and final_state["critique"] is not None
    report_md = render_report(final_state["draft"], registry)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        stat_card("Revisions used", f"{final_state['revision_count']} / {MAX_REVISIONS}"),
        unsafe_allow_html=True,
    )
    c2.markdown(stat_card("Citations", len(final_state["citations"])), unsafe_allow_html=True)
    c3.markdown(stat_card("Sources retrieved", len(registry)), unsafe_allow_html=True)
    c4.markdown(
        stat_card("Outcome", "Force-approved" if forced else "Approved"),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    tab_report, tab_trace, tab_sources = st.tabs(["Final report", "Trace", "Sources"])

    with tab_report:
        with st.container(border=True):
            st.markdown(report_md)
        st.download_button("Download report (.md)", report_md, file_name="research_report.md")

    with tab_trace:
        st.caption(
            "S-numbers in this tab are internal source IDs. "
            "The Sources tab maps each one to its title and URL."
        )

        st.markdown("#### Sub-questions")
        for i, q in enumerate(final_state["sub_questions"], 1):
            st.markdown(f"{i}. {q}")

        st.markdown("#### Tool calls and sources retrieved")
        for f in final_state["findings"]:
            with st.expander(f["sub_question"]):
                for s in f["sources"]:
                    tool = s.get("tool", "unknown")
                    where = s["url"] if s["url"] else "local file"
                    st.markdown(f"- `{tool}`: **{s['title']}** ({where})")

        st.markdown("#### Drafts and critic feedback")
        for i, draft in enumerate(trace["drafts"]):
            verdict = trace["critiques"][i] if i < len(trace["critiques"]) else None
            label = f"Draft {i + 1}"
            kind, text = None, None
            if verdict:
                if verdict["forced"]:
                    kind, text = "warn", "Force-approved"
                elif verdict["approved"]:
                    kind, text = "ok", "Approved"
                else:
                    kind, text = "bad", "Rejected"
                label += f" ({text})"
            with st.expander(label):
                if kind:
                    st.markdown(pill(text, kind), unsafe_allow_html=True)
                st.markdown(link_citations(draft, registry))
                if verdict and verdict["critique"]:
                    st.markdown("**Critic feedback**")
                    st.markdown(humanize(verdict["critique"], registry))

    with tab_sources:
        rows = [
            {
                "ID": sid,
                "Type": "Web" if src["url"] else "Local file",
                "Title": src["title"],
                "URL": src["url"],
                "Answers": src["sub_question"],
            }
            for sid, src in registry.items()
        ]
        st.dataframe(
            rows,
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("URL")},
        )