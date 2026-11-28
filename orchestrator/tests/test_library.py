"""The write path into Spotter's study library. No network, no database: PubMed, Postgres
and Supabase are all faked, and nothing here can reach the production library."""

import json

import httpx
import pytest

from orchestrator import cli, library, tools

ART = {"pmid": "12345678", "title": "Creatine and kidney function", "journal": "J Nephrol", "year": "2021", "types": ["Meta-Analysis"], "abstract": "x"}
# the row shape Spotter's add_studies(rows jsonb) RPC reads (supabase/ask.sql in the Spotter repo)
ROW = {"pmid": "12345678", "title": "Creatine and kidney function", "journal": "J Nephrol", "year": 2021, "types": ["Meta-Analysis"], "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"}


@pytest.fixture
def fakes(monkeypatch):
    saved, posts = [], []
    monkeypatch.setattr(tools, "fetch_articles", lambda pmids: [ART] if "12345678" in pmids else [])
    monkeypatch.setattr("orchestrator.db.save_studies", lambda rows, note: saved.append((rows, note)) or len(rows))

    def handler(req: httpx.Request):
        posts.append(req)
        return httpx.Response(204)

    real = httpx.Client
    monkeypatch.setattr(library.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler)))
    for k in ("LIBRARY_TARGET", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "LIBRARY_LOG"):
        monkeypatch.delenv(k, raising=False)
    return saved, posts


def test_rows_match_spotters_studies_shape():
    assert library.rows([ART]) == [ROW]
    assert library.rows([{**ART, "year": None, "journal": None, "types": None}])[0] == {**ROW, "year": None, "journal": "", "types": []}


def test_default_target_stays_local(fakes):
    saved, posts = fakes
    out = tools.save_to_library(tools.SaveStudies(pmids=["12345678"], note="strong MA"))
    assert out == "Saved 1 studies to the local library: 12345678."
    assert saved == [([ROW], "strong MA")] and posts == []


def test_spotter_target_calls_add_studies_rpc(fakes, monkeypatch):
    saved, posts = fakes
    monkeypatch.setenv("LIBRARY_TARGET", "spotter")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    out = tools.save_to_library(tools.SaveStudies(pmids=["12345678"], note="n"))
    assert out.startswith("Saved 1 studies to Spotter's library")
    [req] = posts
    assert str(req.url) == "https://example.supabase.co/rest/v1/rpc/add_studies"
    assert req.headers["apikey"] == "test-key" and req.headers["authorization"] == "Bearer test-key"
    assert json.loads(req.content) == {"rows": [ROW]}
    assert saved  # the local audit copy (with the note) is written too


def test_spotter_target_without_credentials_fails_loudly(fakes, monkeypatch):
    monkeypatch.setenv("LIBRARY_TARGET", "spotter")
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        tools.save_to_library(tools.SaveStudies(pmids=["12345678"], note="n"))


def test_pmids_missing_from_pubmed_are_not_saved(fakes):
    saved, posts = fakes
    assert "Nothing saved" in tools.save_to_library(tools.SaveStudies(pmids=["99999999"], note="n"))
    assert saved == [] and posts == []


def test_cli_review_sends_the_human_decision():
    sent = []
    pending = {
        "id": "r1",
        "goal": "Does creatine harm kidneys?\nResearch this on PubMed",
        "status": "waiting_approval",
        "pending_approval": {
            "tool": "save_to_library",
            "input": {"pmids": ["12345678"], "note": "n"},
            "studies": [{"pmid": "12345678", "title": "Creatine and kidney function"}],
        },
    }

    def handler(req: httpx.Request):
        if req.method == "POST":
            sent.append((req.url.path, json.loads(req.content)))
            return httpx.Response(202, json={})
        if req.url.path == "/runs":
            waiting = req.url.params["status"] == "waiting_approval" and not sent
            return httpx.Response(200, json=[pending] if waiting else [])
        return httpx.Response(200, json=pending)

    c = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))
    answers = iter(["n", "only meta-analyses"])
    assert cli.review(c, ask=lambda _: next(answers)) == 1
    assert sent == [("/runs/r1/approve", {"approved": False, "reason": "only meta-analyses"})]
    assert "PMID 12345678  Creatine and kidney function" in cli.show(pending)


def test_seed_file_skips_comments(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# c\nQ one?\n\n  Q two?\n", encoding="utf-8")
    assert cli.questions(str(f)) == ["Q one?", "Q two?"]
    assert len(cli.questions("seeds.txt")) >= 8


def test_run_local_researches_waits_and_saves_on_approval(fakes, monkeypatch, tmp_path):
    from dataclasses import replace

    from orchestrator.tests.test_graph import READ, READ_OUT, FakeLLM, done, more

    saved, posts = fakes
    log = tmp_path / "library.jsonl"
    monkeypatch.setenv("LIBRARY_LOG", str(log))
    monkeypatch.setitem(tools.TOOLS, "read_abstracts", replace(tools.TOOLS["read_abstracts"], fn=lambda a: READ_OUT))
    seeds = tmp_path / "s.txt"
    seeds.write_text("Does creatine harm kidneys?\n", encoding="utf-8")
    save = ("", "save_to_library", {"pmids": ["12345678"], "note": "meta-analysis, no harm"})
    llm = FakeLLM([READ, save], [more(), done("No [PMID:12345678].")])
    prompts = []
    [res] = cli.run_local(str(seeds), llm=llm, ask=lambda p: prompts.append(p) or "y")
    assert prompts == ["Approve? [y/N] "]  # nothing is written before a human says yes
    assert res["answer"] == "No [PMID:12345678]."
    assert [json.loads(x) for x in log.read_text().splitlines()] == [{**ROW, "note": "meta-analysis, no harm"}]
    assert saved == [] and posts == []  # local log only: no Postgres, no Supabase
