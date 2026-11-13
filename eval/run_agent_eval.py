"""
Measures how well the research agent keeps its citations honest.

For every goal it runs the full graph and audits each draft the Writer produces:
  - how many [S#] citation markers the draft contains,
  - how many of those point to a source ID that was never retrieved (fabricated),
  - whether the Critic approved or rejected that draft.

Headline numbers it prints:
  - fabricated citations caught by the parser, out of all markers checked
  - fabricated citations that survived into a final report
  - how often the Critic rejected a first draft, and how often the revision loop fixed it

Results are cached per goal in a JSONL file, so a rate-limit stop is fine: rerun and it
continues. Use --limit to cap new goals per session and protect your daily token budget.

Usage (from the project root):
    python -m eval.run_agent_eval --limit 8
    python -m eval.run_agent_eval --summary-only
"""
import argparse
import json
import time
from pathlib import Path

from graph.build_graph import build_graph
from graph.nodes.writer import build_source_registry, extract_citations

GOALS_PATH = Path("eval/goals.json")
RESULTS_PATH = Path("eval/results/agent_runs.jsonl")


def read_rows():
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_draft(draft, registry):
    """Counts valid and fabricated citation markers in one draft."""
    citations, invalid_ids = extract_citations(draft, registry)
    return {
        "markers": len(citations) + len(invalid_ids),
        "valid": len(citations),
        "invalid_markers": len(invalid_ids),
        "invalid_ids": sorted(set(invalid_ids)),
    }


def run_goal(graph, goal_id, goal):
    initial_state = {
        "question": goal, "sub_questions": [], "findings": [], "draft": "",
        "citations": [], "critique": None, "revision_count": 0, "approved": False,
    }
    registry, drafts, verdicts, final = {}, [], [], initial_state
    started = time.time()

    for step in graph.stream(initial_state, stream_mode="updates"):
        for node, update in step.items():
            final = update
            if node == "researcher":
                registry, _ = build_source_registry(update["findings"])
            elif node == "writer":
                drafts.append(audit_draft(update["draft"], registry))
            elif node == "critic":
                forced = bool(update["approved"] and update["critique"] is not None)
                verdicts.append({"approved": bool(update["approved"]), "forced": forced})

    for draft, verdict in zip(drafts, verdicts):
        draft["critic_approved"] = verdict["approved"]

    return {
        "id": goal_id,
        "goal": goal,
        "sources_retrieved": len(registry),
        "drafts": drafts,
        "verdicts": verdicts,
        "revisions": final["revision_count"],
        "force_approved": bool(verdicts and verdicts[-1]["forced"]),
        "seconds": round(time.time() - started, 1),
    }


def summarize(rows):
    runs = len(rows)
    all_drafts = [d for r in rows for d in r["drafts"]]
    markers = sum(d["markers"] for d in all_drafts)
    fabricated = sum(d["invalid_markers"] for d in all_drafts)
    finals = [r["drafts"][-1] for r in rows if r["drafts"]]
    final_markers = sum(d["markers"] for d in finals)
    final_fabricated = sum(d["invalid_markers"] for d in finals)

    first_rejected = [r for r in rows if r["verdicts"] and not r["verdicts"][0]["approved"]]
    recovered = [r for r in first_rejected if r["verdicts"][-1]["approved"] and not r["force_approved"]]

    return {
        "runs": runs,
        "drafts_audited": len(all_drafts),
        "citation_markers_checked": markers,
        "fabricated_markers_caught": fabricated,
        "drafts_with_fabricated_ids": sum(1 for d in all_drafts if d["invalid_markers"] > 0),
        "final_report_markers": final_markers,
        "fabricated_in_final_reports": final_fabricated,
        "first_drafts_rejected": len(first_rejected),
        "rejected_then_fixed": len(recovered),
        "force_approved_runs": sum(1 for r in rows if r["force_approved"]),
        "mean_revisions": sum(r["revisions"] for r in rows) / runs if runs else None,
    }


def print_summary(rows):
    if not rows:
        raise SystemExit("No completed runs yet.")
    s = summarize(rows)
    print(f"\n=== Agent evaluation: {s['runs']} runs, {s['drafts_audited']} drafts ===")
    print(f"Citation markers checked:           {s['citation_markers_checked']}")
    print(f"Fabricated IDs caught by parser:    {s['fabricated_markers_caught']} "
          f"({s['drafts_with_fabricated_ids']} drafts affected)")
    print(f"Fabricated IDs in final reports:    {s['fabricated_in_final_reports']} "
          f"of {s['final_report_markers']} final-report citations")
    print(f"First drafts rejected by Critic:    {s['first_drafts_rejected']} of {s['runs']}")
    print(f"  ...of those, fixed by revision:   {s['rejected_then_fixed']}")
    print(f"Force-approved at the cap:          {s['force_approved_runs']}")
    print(f"Mean revisions per run:             {s['mean_revisions']:.2f}")
    print("\nReport whatever these say, including zeros. 'Zero fabricated IDs reached a final report "
          "across N runs' is a fair claim only if the final-report line shows 0.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max new goals this session (0 = all remaining)")
    ap.add_argument("--sleep", type=float, default=5.0, help="seconds between runs")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not args.summary_only:
        goals = json.loads(GOALS_PATH.read_text(encoding="utf-8"))
        done = {r["id"] for r in read_rows()}
        todo = [g for g in goals if g["id"] not in done]
        if args.limit:
            todo = todo[:args.limit]

        if todo:
            graph = build_graph()
            for i, g in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] {g['id']}: {g['goal']}")
                try:
                    row = run_goal(graph, g["id"], g["goal"])
                except Exception as exc:
                    print(f"  failed (rerun later to retry): {exc}")
                    continue
                RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with RESULTS_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                time.sleep(args.sleep)
        else:
            print("All goals already evaluated.")

    print_summary(read_rows())


if __name__ == "__main__":
    main()
