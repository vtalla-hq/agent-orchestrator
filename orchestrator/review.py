"""Rule-based reviewer for save_to_library, used by `cli run --auto` in place of a person.

It applies the bar a careful reviewer would, from PubMed's own metadata rather than the
agent's say-so: each study must exist on PubMed, be strong evidence, not be retracted or
an opinion piece, and not be animal-only. The agent has already guaranteed it read every
PMID it proposes (graph.py). Studies that fail are dropped with a reason; the rest are saved.
"""

import re

from .tools import fetch_articles

STRONG = {
    "Meta-Analysis",
    "Systematic Review",
    "Randomized Controlled Trial",
    "Controlled Clinical Trial",
    "Clinical Trial",
    "Clinical Trial, Phase III",
    "Clinical Trial, Phase IV",
    "Practice Guideline",
    "Guideline",
    "Consensus Development Conference",
    "Observational Study",
}
BAD = {
    "Retracted Publication",
    "Retraction of Publication",
    "Expression of Concern",
    "Case Reports",
    "Comment",
    "Letter",
    "Editorial",
    "Published Erratum",
    "Preprint",
}
# brand-new articles often have no publication types yet; fall back to the title
STRONG_TITLE = re.compile(r"meta-?analys|systematic review|umbrella review|randomi[sz]ed|placebo-controlled|controlled trial|cohort|position stand|guideline|consensus", re.I)


def judge(a: dict) -> str | None:
    """None if the study passes, else why not."""
    types = set(a.get("types") or [])
    if bad := types & BAD:
        return ", ".join(sorted(bad))
    mesh = set(a.get("mesh") or [])
    if "Animals" in mesh and "Humans" not in mesh:
        return "animal study"
    if types & STRONG or STRONG_TITLE.search(a.get("title", "")):
        return None
    return f"not strong evidence ({', '.join(sorted(types)) or 'no study type'})"


def review(pending: dict, fetch=fetch_articles) -> dict:
    """A resume value for the save_to_library interrupt."""
    pmids = pending["input"]["pmids"]
    found = {a["pmid"]: a for a in fetch(pmids)}
    kept, why = [], {}
    for p in pmids:
        reason = judge(found[p]) if p in found else "not on PubMed"
        if reason:
            why[p] = reason
        else:
            kept.append(p)
    dropped = "; ".join(f"{p}: {r}" for p, r in why.items())
    if not kept:
        return {"approved": False, "by": "The auto-reviewer", "reason": f"none met the evidence bar ({dropped}). Find meta-analyses, RCTs or guidelines instead."}
    return {"approved": True, "by": "The auto-reviewer", "pmids": kept, "reason": f"dropped {dropped}" if why else ""}
