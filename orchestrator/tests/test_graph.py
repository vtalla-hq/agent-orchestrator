from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator import graph as G
from orchestrator.tools import TOOLS, parse_articles, public_url


class FakeLLM:
    """Plays back scripted answers: tool calls for act, schema instances for plan/reflect."""

    def __init__(self, calls, reflections):
        self.calls, self.reflections = list(calls), list(reflections)
        self.tool_prompts = []

    def structured(self, system, prompt, schema):
        if schema is G.Plan:
            return G.Plan(steps=["search", "read", "answer"])
        return self.reflections.pop(0)

    def choose_tool(self, system, prompt, tools):
        self.tool_prompts.append(prompt)
        return self.calls.pop(0)


def done(answer="ok"):
    return G.Reflection(critique="enough", done=True, replan=False, answer=answer)


def more():
    return G.Reflection(critique="need more", done=False, replan=False, answer="")


def stub(monkeypatch, name, out="stubbed"):
    seen = []
    monkeypatch.setitem(TOOLS, name, replace(TOOLS[name], fn=lambda a: seen.append(a) or out))
    return seen


def run(llm, payload, thread="t1", logs=None, cp=None):
    cp = cp or InMemorySaver()
    g = G.build_graph(llm, (lambda *a, **k: logs.append((a[1], a[2]))) if logs is not None else (lambda *a, **k: None), cp)
    cfg = {"configurable": {"thread_id": thread}}
    g.invoke(payload, cfg)
    return g, cfg


def test_plan_act_reflect_loop_and_log(monkeypatch):
    seen = stub(monkeypatch, "search_pubmed", "PMID 1 | 2020 | J | RCT | title")
    logs = []
    llm = FakeLLM([("", "search_pubmed", {"query": "creatine kidney"}), ("", None, {})], [more(), done("Creatine is fine [PMID:1]")])
    g, cfg = run(llm, {"goal": "Is creatine bad for kidneys?"}, logs=logs)
    st = g.get_state(cfg).values
    assert st["answer"] == "Creatine is fine [PMID:1]"
    assert seen[0].query == "creatine kidney" and seen[0].max_results == 8  # validated + defaults applied
    assert st["step"] == 1 and st["iterations"] == 2
    assert [n for n, _ in logs] == ["plan", "act", "tools", "reflect", "act", "tools", "reflect"]
    assert "PMID 1" in llm.tool_prompts[1]  # memory from step 1 reaches step 2


def test_invalid_input_becomes_observation(monkeypatch):
    seen = stub(monkeypatch, "read_abstracts")
    llm = FakeLLM([("", "read_abstracts", {"pmids": ["1", "2", "3", "4"]})], [done()])
    g, cfg = run(llm, {"goal": "x"})
    assert not seen
    assert "Invalid input" in g.get_state(cfg).values["observations"][0]["output"]


READ = ("", "read_abstracts", {"pmids": ["12345678"]})
READ_OUT = "PMID 12345678 (2021, J Nephrol)\nCreatine and kidney function: a meta-analysis\nNo harm."


@pytest.mark.parametrize("approved", [True, False])
def test_high_risk_tool_waits_for_human(monkeypatch, approved):
    stub(monkeypatch, "read_abstracts", READ_OUT)
    seen = stub(monkeypatch, "save_to_library", "Saved 1 studies to the local library: 12345678.")
    logs = []
    save = ("worth keeping", "save_to_library", {"pmids": ["12345678", "87654321"], "note": "good meta-analysis"})
    llm = FakeLLM([READ, save], [more(), done()])
    cp = InMemorySaver()
    g, cfg = run(llm, {"goal": "save it"}, logs=logs, cp=cp)
    st = g.get_state(cfg)
    assert st.next == ("tools",) and not seen
    # the unread PMID is dropped before a human sees it; the read one comes with its title
    assert st.interrupts[0].value == {
        "tool": "save_to_library",
        "input": {"pmids": ["12345678"], "note": "good meta-analysis"},
        "why": "worth keeping",
        "studies": [{"pmid": "12345678", "title": "Creatine and kidney function: a meta-analysis"}],
        "dropped_unread": ["87654321"],
    }
    assert ("tools", "interrupted") in logs

    g.invoke(Command(resume={"approved": approved, "reason": "dup"}), cfg)
    out = g.get_state(cfg).values["observations"][-1]["output"]
    assert [a.pmids for a in seen] == ([["12345678"]] if approved else [])
    assert out.startswith("Saved") if approved else "rejected" in out


def test_save_without_reading_is_refused_without_asking(monkeypatch):
    seen = stub(monkeypatch, "save_to_library")
    g, cfg = run(FakeLLM([("", "save_to_library", {"pmids": ["12345678"], "note": "n"})], [done()]), {"goal": "x"})
    st = g.get_state(cfg)
    assert not st.interrupts and not seen
    assert "read the abstracts of 12345678" in st.values["observations"][0]["output"]


def test_save_rejects_malformed_pmids(monkeypatch):
    seen = stub(monkeypatch, "save_to_library")
    g, cfg = run(FakeLLM([("", "save_to_library", {"pmids": ["PMID:1"], "note": "n"})], [done()]), {"goal": "x"})
    assert not seen and "Invalid input" in g.get_state(cfg).values["observations"][0]["output"]


def test_iteration_cap_forces_an_answer(monkeypatch):
    stub(monkeypatch, "search_pubmed")
    n = G.MAX_ITERATIONS
    llm = FakeLLM([("", "search_pubmed", {"query": "q"})] * n, [more()] * n)
    g, cfg = run(llm, {"goal": "x"})
    st = g.get_state(cfg).values
    assert st["iterations"] == n and st["answer"] == "need more"


def test_tool_crash_is_an_observation(monkeypatch):
    def boom(a):
        raise TimeoutError("slow")

    monkeypatch.setitem(TOOLS, "search_pubmed", replace(TOOLS["search_pubmed"], fn=boom))
    g, cfg = run(FakeLLM([("", "search_pubmed", {"query": "q"})], [done()]), {"goal": "x"})
    assert "Tool failed: TimeoutError" in g.get_state(cfg).values["observations"][0]["output"]


def test_every_tool_has_a_typed_schema():
    for t in TOOLS.values():
        assert t.json_schema()["type"] == "object" and t.json_schema()["properties"]
    assert [t.name for t in TOOLS.values() if t.high_risk] == ["save_to_library"]


def test_public_url_blocks_internal_targets():
    for bad in ["http://localhost/", "http://127.0.0.1:6379/", "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://x.org"]:
        assert not public_url(bad), bad


def test_parse_articles():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
    <Journal><Title>Journal of X</Title><ISOAbbreviation>J X</ISOAbbreviation><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Creatine and <i>kidneys</i></ArticleTitle>
    <Abstract><AbstractText Label="METHODS">RCT.</AbstractText><AbstractText Label="RESULTS">No harm.</AbstractText></Abstract>
    <PublicationTypeList><PublicationType>Journal Article</PublicationType><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [a] = parse_articles(xml)
    assert a == {"pmid": "42", "title": "Creatine and kidneys", "journal": "J X", "year": "2021", "types": ["Randomized Controlled Trial"], "mesh": [], "abstract": "METHODS: RCT. RESULTS: No harm."}


def test_passages_around_keyword():
    from orchestrator.tools import passages

    text = "nav " * 2000 + "Creatine is safe at usual intakes. " + "x " * 2000 + "creatine and kidneys."
    out = passages(text, "creatine", radius=20)
    assert "Creatine is safe" in out and "creatine and kidneys" in out and " … " in out and len(out) < 200
    assert passages(text, "zzz").endswith("not found on the page.")
    assert passages("abc", None) == "abc"
    first = passages("a" * 5000, None)
    assert first.endswith("call again with start=3000 for more]") and passages("a" * 5000, None, start=3000) == "a" * 2000


def test_citations_to_unread_studies_are_dropped():
    obs = [
        {"tool": "search_pubmed", "input": {}, "output": "PMID 22222222 | 2020 | J | RCT | t"},
        {"tool": "read_abstracts", "input": {}, "output": "PMID 11111111 (2021, J)\ntitle\nabstract"},
    ]
    out = G.clean_citations("Safe [PMID:11111111]. Also fine [PMID:22222222], see PMID 11111111.", obs)
    assert out == "Safe [PMID:11111111]. Also fine, see [PMID:11111111]."


def test_run_cannot_finish_without_a_required_tool(monkeypatch):
    # seen live: the model answered "Saved to Spotter's library" without ever calling the tool
    stub(monkeypatch, "read_abstracts", READ_OUT)
    llm = FakeLLM([READ, ("", "save_to_library", {"pmids": ["12345678"], "note": "n"})], [done("Saved it."), done()])
    g, cfg = run(llm, {"goal": "research and save", "require": ["save_to_library"]})
    st = g.get_state(cfg)
    assert st.interrupts and st.interrupts[0].value["tool"] == "save_to_library"
    assert "call save_to_library now" in llm.tool_prompts[1]  # the reviewer's note reaches act
    g.invoke(Command(resume={"approved": False, "reason": "no"}), cfg)
    assert g.get_state(cfg).values["answer"] == "ok"  # a rejection counts as attempted
