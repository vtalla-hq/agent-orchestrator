"""Library builder CLI, talking to the API (docker compose up).

    python -m orchestrator.cli seed [seeds.txt]   queue one research run per question
    python -m orchestrator.cli review [--wait]    approve or reject each pending save_to_library
    python -m orchestrator.cli run [seeds.txt]    the same in one process, no Docker (asks for approval inline)

Approval stays human: `review` shows the studies the agent read and wants to save,
and you answer y / n (with a reason, which goes back to the agent).
"""

import argparse
import os
import time

import httpx

API = os.getenv("ORCHESTRATOR_API", "http://localhost:8000")

GOAL = (
    "{q}\nResearch this on PubMed: search, read the abstracts of the strongest 2-4 relevant studies "
    "(meta-analyses, guidelines, RCTs, large cohorts), answer with citations, then save the best of the "
    "studies you read to Spotter's library with a one-line note on why each matters."
)


def questions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def seed(c: httpx.Client, path: str) -> list[str]:
    ids = []
    for q in questions(path):
        r = c.post("/runs", json={"goal": GOAL.format(q=q), "require": ["save_to_library"]})
        r.raise_for_status()
        ids.append(r.json()["id"])
        print(f"queued {ids[-1]}  {q}")
    return ids


def show(run: dict) -> str:
    p = run.get("pending_approval") or {}
    lines = [f"\nRun {run['id']}", f"Goal: {run['goal'].splitlines()[0]}", f"Wants to save to Spotter's library ({p.get('tool')}):"]
    for s in p.get("studies") or [{"pmid": x, "title": ""} for x in p.get("input", {}).get("pmids", [])]:
        lines.append(f"  PMID {s['pmid']}  {s['title']}\n    https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/")
    if p.get("dropped_unread"):
        lines.append(f"  (dropped, not read by the agent: {', '.join(p['dropped_unread'])})")
    lines.append(f"Note: {p.get('input', {}).get('note', '')}")
    if p.get("why"):
        lines.append(f"Agent says: {p['why'][:400]}")
    return "\n".join(lines)


def review(c: httpx.Client, wait: bool = False, ask=input, poll: float = 10) -> int:
    decided = 0
    while True:
        pending = c.get("/runs", params={"status": "waiting_approval"}).json()
        for r in pending:
            run = c.get(f"/runs/{r['id']}").json()
            print(show(run))
            a = ask("Approve? [y/N/s=skip] ").strip().lower()
            if a == "s":
                continue
            ok = a in ("y", "yes")
            reason = "" if ok else ask("Reason (sent to the agent): ").strip()
            c.post(f"/runs/{r['id']}/approve", json={"approved": ok, "reason": reason}).raise_for_status()
            decided += 1
        busy = [r for s in ("queued", "running") for r in c.get("/runs", params={"status": s}).json()]
        if not wait or (not busy and not pending):
            return decided
        time.sleep(poll)


def run_local(path: str, llm=None, ask=input, auto: bool = False, reviewer=None) -> list[dict]:
    """Research each question in this process: no Redis, Postgres or Celery. State is in memory,
    approvals are asked here, and approved saves are logged to LIBRARY_LOG (default library.jsonl)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from .graph import build_graph
    from .llm import get_llm

    os.environ.setdefault("LIBRARY_LOG", "library.jsonl")
    g = build_graph(llm or get_llm(), checkpointer=InMemorySaver())
    results = []
    for i, q in enumerate(questions(path)):
        cfg = {"configurable": {"thread_id": f"local-{i}"}}
        print(f"\n=== {q}")
        try:
            g.invoke({"goal": GOAL.format(q=q), "require": ["save_to_library"]}, cfg)
            while (st := g.get_state(cfg)).interrupts:
                pending = st.interrupts[0].value
                print(show({"id": f"local-{i}", "goal": q, "pending_approval": pending}))
                if auto:
                    from .review import review

                    decision = (reviewer or review)(pending)
                    print(f"Auto-review: {'approved ' + ', '.join(decision['pmids']) if decision['approved'] else 'rejected'}. {decision['reason']}")
                else:
                    ok = ask("Approve? [y/N] ").strip().lower() in ("y", "yes")
                    decision = {"approved": ok, "reason": "" if ok else ask("Reason (sent to the agent): ").strip()}
                g.invoke(Command(resume=decision), cfg)
        except Exception as e:  # a rate limit or model hiccup on one question shouldn't end the batch
            print(f"\nSkipped after an error: {type(e).__name__}: {str(e)[:200]}")
            results.append({"question": q, "answer": None, "error": str(e)})
            continue
        answer = g.get_state(cfg).values.get("answer")
        print(f"\nAnswer: {answer}")
        results.append({"question": q, "answer": answer})
    return results


def load_env(path: str = ".env") -> None:
    """KEY=value lines from .env into the environment (docker compose does this for the API; `run` has no compose)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for ln in f:
            k, sep, v = ln.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m orchestrator.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("file", nargs="?", default="seeds.txt")
    rv = sub.add_parser("review")
    rv.add_argument("--wait", action="store_true", help="keep polling until no run is queued, running or waiting")
    lo = sub.add_parser("run", help="research in this process, no Docker needed")
    lo.add_argument("file", nargs="?", default="seeds.txt")
    lo.add_argument("--auto", action="store_true", help="a rule-based reviewer approves saves instead of you (see orchestrator/review.py)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        load_env()
        if os.getenv("LIBRARY_TARGET") == "spotter" and not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            raise SystemExit("LIBRARY_TARGET=spotter but SUPABASE_SERVICE_ROLE_KEY is empty in .env. Paste the sb_secret_ key there first.")
        print(f"Approved studies go to: {'Spotter (production library)' if os.getenv('LIBRARY_TARGET') == 'spotter' else os.getenv('LIBRARY_LOG', 'library.jsonl')}")
        run_local(a.file, auto=a.auto)
        return
    with httpx.Client(base_url=API, timeout=30) as c:
        if a.cmd == "seed":
            seed(c, a.file)
        else:
            print(f"{review(c, a.wait)} decisions sent.")


if __name__ == "__main__":
    main()
