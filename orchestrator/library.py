"""Approved studies -> Spotter's shared study library.

Spotter keeps the library in Supabase (`public.studies`, see supabase/ask.sql in the
Spotter repo) and writes to it only through the `add_studies(rows jsonb)` RPC with the
service key. This module writes the exact row shape that RPC takes:

    {pmid, title, journal, year, types, url}

so an approved save looks the same as a study Spotter's own Ask cited. Spotter has no
column for the reviewer's note, so the note (and every approval) is kept in the local
Postgres `library` table as an audit trail.

Writes go to Spotter only with LIBRARY_TARGET=spotter plus SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY. The default, LIBRARY_TARGET=local, never leaves this machine.
"""

import json
import os

import httpx

PUBMED = "https://pubmed.ncbi.nlm.nih.gov/{}/"


def rows(articles: list[dict]) -> list[dict]:
    """PubMed articles -> rows for Spotter's add_studies RPC."""
    out = []
    for a in articles:
        year = a.get("year")
        out.append(
            {
                "pmid": a["pmid"],
                "title": a["title"],
                "journal": a.get("journal") or "",
                "year": int(year) if year and str(year).isdigit() else None,
                "types": list(a.get("types") or []),
                "url": PUBMED.format(a["pmid"]),
            }
        )
    return out


def record(rs: list[dict], note: str) -> None:
    """Keep the approval (with its note) locally: Postgres, or a JSONL file when LIBRARY_LOG is set (cli run)."""
    if path := os.getenv("LIBRARY_LOG"):
        with open(path, "a", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({**r, "note": note}) + "\n")
        return
    from . import db

    db.save_studies(rs, note)


def target() -> str:
    return os.getenv("LIBRARY_TARGET", "local")


def push_to_spotter(rs: list[dict], client: httpx.Client | None = None) -> None:
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("LIBRARY_TARGET=spotter needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    c = client or httpx.Client(timeout=20)
    r = c.post(
        f"{url}/rest/v1/rpc/add_studies",
        headers={"apikey": key, "authorization": f"Bearer {key}", "content-type": "application/json"},
        json={"rows": rs},
    )
    r.raise_for_status()
