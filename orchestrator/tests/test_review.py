"""The rule-based reviewer that stands in for a person with `cli run --auto`."""

import json
from dataclasses import replace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator import cli, graph as G, tools
from orchestrator.review import judge, review
from orchestrator.tests.test_graph import READ, READ_OUT, FakeLLM, done, more, stub


def art(pmid="12345678", title="Creatine and kidneys", types=(), mesh=("Humans",)):
    return {"pmid": pmid, "title": title, "journal": "J", "year": "2020", "types": list(types), "mesh": list(mesh)}


def test_judge_applies_the_evidence_bar():
    assert judge(art(types=["Meta-Analysis"])) is None
    assert judge(art(types=["Randomized Controlled Trial"])) is None
    assert judge(art(title="Creatine and kidney function: a systematic review and meta-analysis")) is None  # no types yet
    assert judge(art(types=["Meta-Analysis", "Retracted Publication"])) == "Retracted Publication"
    assert judge(art(types=["Case Reports"])) == "Case Reports"
    assert judge(art(types=["Randomized Controlled Trial"], mesh=["Animals", "Rats"])) == "animal study"
    assert judge(art(types=["Review"])).startswith("not strong evidence")


def test_review_keeps_the_strong_ones_and_says_why_it_dropped_the_rest():
    arts = [art("11111111", types=["Meta-Analysis"]), art("22222222", types=["Letter"])]
    d = review({"input": {"pmids": ["11111111", "22222222", "33333333"]}}, fetch=lambda p: arts)
    assert d["approved"] and d["pmids"] == ["11111111"]
    assert "22222222: Letter" in d["reason"] and "33333333: not on PubMed" in d["reason"]
    d = review({"input": {"pmids": ["22222222"]}}, fetch=lambda p: arts)
    assert not d["approved"] and "none met the evidence bar" in d["reason"]


def test_partial_approval_saves_only_the_approved_subset(monkeypatch):
    two = READ_OUT + "\n\n---\n\nPMID 87654321 (2020, J)\nA case report\nx"
    stub(monkeypatch, "read_abstracts", two)
    seen = stub(monkeypatch, "save_to_library", "Saved 1 studies.")
    llm = FakeLLM([READ, ("", "save_to_library", {"pmids": ["12345678", "87654321"], "note": "n"})], [more(), done()])
    g = G.build_graph(llm, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "p"}}
    g.invoke({"goal": "x"}, cfg)
    g.invoke(Command(resume={"approved": True, "by": "The auto-reviewer", "pmids": ["12345678"], "reason": "dropped 87654321: Case Reports"}), cfg)
    assert [a.pmids for a in seen] == [["12345678"]]
    assert g.get_state(cfg).values["observations"][-1]["output"].endswith("The auto-reviewer: dropped 87654321: Case Reports")


def test_cli_auto_mode_never_asks_and_writes_what_the_reviewer_passed(monkeypatch, tmp_path):
    log = tmp_path / "library.jsonl"
    monkeypatch.setenv("LIBRARY_LOG", str(log))
    monkeypatch.delenv("LIBRARY_TARGET", raising=False)
    monkeypatch.setitem(tools.TOOLS, "read_abstracts", replace(tools.TOOLS["read_abstracts"], fn=lambda a: READ_OUT))
    meta = {"pmid": "12345678", "title": "Creatine and kidney function: a meta-analysis", "journal": "J", "year": "2021", "types": ["Meta-Analysis"], "mesh": ["Humans"]}
    monkeypatch.setattr(tools, "fetch_articles", lambda pmids: [meta])
    seeds = tmp_path / "s.txt"
    seeds.write_text("Q?\n", encoding="utf-8")
    llm = FakeLLM([READ, ("", "save_to_library", {"pmids": ["12345678"], "note": "n"})], [more(), done()])

    def no_human(_):
        raise AssertionError("auto mode asked a human")

    [res] = cli.run_local(str(seeds), llm=llm, ask=no_human, auto=True, reviewer=lambda p: review(p, fetch=lambda x: [meta]))
    assert res["answer"] == "ok"
    assert [json.loads(x)["pmid"] for x in log.read_text().splitlines()] == ["12345678"]
