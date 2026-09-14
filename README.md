# ResearchAgent

Autonomous multi-agent research system that takes a goal, plans sub-questions,
retrieves and synthesizes sources, then **critiques and revises its own output**
until it clears a quality bar — instead of returning a single unchecked draft.

**[Live Demo →](#)** *(add your Streamlit Cloud link after deployment)*

---

## Why this exists

Most "AI research assistant" demos are a single LLM call with a web search tool
bolted on. That produces a draft, not a verified answer — there's no mechanism
that catches an ungrounded claim or an incomplete answer before it reaches you.

ResearchAgent adds a second agent whose only job is to grade the first agent's
work against a fixed rubric (task completion, groundedness, clarity, redundancy)
and force a revision loop until the report actually clears the bar. This is the
same idea Chain-of-Thought self-critique / reflection papers describe, applied
to a concrete, measurable task.

## Architecture

```
   Goal
    │
    ▼
┌─────────┐     ┌───────────┐     ┌─────────┐     ┌────────┐     ┌──────────┐
│ Planner │────▶│ Retriever │────▶│ Drafter │────▶│ Critic │────▶│ Reviser  │
└─────────┘     └───────────┘     └─────────┘     └───┬────┘     └────┬─────┘
                                                        │  fail        │
                                                        └──────────────┘
                                                        pass ▼
                                                   Final Report
```

| Stage | Responsibility |
|---|---|
| **Planner** | Breaks the goal into 3-5 independently searchable sub-questions |
| **Retriever** | Web-searches each sub-question, returns attributed snippets |
| **Drafter** | Synthesizes sources into a structured report with inline citation tags |
| **Critic** | Scores the draft 0-10 on 4 dimensions, returns specific revision instructions |
| **Reviser** | Applies feedback, produces an improved draft |

The critic → reviser loop repeats up to `max_iterations` (default 3) until the
draft passes (`overall ≥ 8` and `groundedness ≥ 8`) or the cap is hit.

## Evaluation

Every claim about this system's quality is backed by a number, not a vibe —
see [`RUBRIC.md`](./RUBRIC.md) for the full methodology and [`benchmark_goals.json`](./benchmark_goals.json)
for the 20-goal test set spanning easy/medium/hard difficulty.

```bash
python scorer.py
```

This produces `eval_results.csv` and an aggregate summary:

```
Task completion rate: XX%
Mean groundedness (self-reported): X.X/10
Mean citation density: X.X tags/100 words
Pass rate within 3 iterations: XX%
Mean iterations to pass/stop: X.X
```
*(Run the benchmark and paste your real numbers here before adding this project to your resume.)*

## Setup

```bash
git clone <your-repo-url>
cd research_agent
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) and either:
- export it: `export GROQ_API_KEY=your_key_here`, or
- paste it into the sidebar when you run the app

## Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI (single goal):**
```bash
python agent.py "Research competitor pricing for AI note-taking apps"
```

**Full benchmark:**
```bash
python scorer.py
```

## Tech stack

- **LLM:** Groq API (Llama 3.3 70B)
- **Search:** DuckDuckGo Search (`ddgs`)
- **UI:** Streamlit
- **Eval:** Custom rubric-based scorer with automated + self-reported groundedness checks

## Project structure

```
research_agent/
├── planner.py          # Goal → sub-questions
├── retriever.py         # Sub-questions → attributed search results
├── drafter.py            # Sources → structured draft with citations
├── critic.py              # Draft → rubric scores + revision instructions
├── reviser.py              # Draft + feedback → improved draft
├── agent.py                 # Orchestrates the full pipeline
├── app.py                    # Streamlit UI
├── scorer.py                  # Benchmark runner
├── benchmark_goals.json        # 20-goal eval set (easy/medium/hard)
├── RUBRIC.md                    # Evaluation methodology
└── requirements.txt
```

## Known limitations

- Web search quality depends on DuckDuckGo's free-tier results; a production
  version would swap in a paid search API (Tavily, Serper) for better recall.
- Critic and drafter currently share the same model — using a stronger model
  for the critic than the drafter is a common pattern worth testing.
- No persistent memory across runs; each goal is evaluated independently.