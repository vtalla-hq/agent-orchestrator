"""Postgres: run status, the step log (for replay and debugging) and the study library."""

import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/orchestrator")

SCHEMA = """
create table if not exists runs (
  id text primary key,
  goal text not null,
  status text not null,           -- queued | running | waiting_approval | done | failed
  answer text,
  error text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
create table if not exists steps (
  id bigserial primary key,
  run_id text not null references runs(id) on delete cascade,
  node text not null,
  status text not null,           -- ok | interrupted | error
  output jsonb,
  error text,
  ms integer,
  created_at timestamptz default now()
);
create index if not exists steps_run on steps(run_id, id);
-- approved saves, in Spotter's studies shape plus the reviewer's note (Spotter has no note column)
create table if not exists library (
  pmid text primary key,
  title text not null,
  journal text,
  year int,
  types text[] not null default '{}',
  url text not null,
  note text,
  added_at timestamptz default now()
);
"""


def conn():
    return psycopg.connect(URL, row_factory=dict_row, autocommit=True)


def setup():
    with conn() as c:
        c.execute(SCHEMA)


def create_run(run_id: str, goal: str):
    with conn() as c:
        c.execute("insert into runs (id, goal, status) values (%s, %s, 'queued')", (run_id, goal))


def update_run(run_id: str, **fields):
    cols = ", ".join(f"{k} = %s" for k in fields)  # keys come from our code only, never user input
    with conn() as c:
        c.execute(f"update runs set {cols}, updated_at = now() where id = %s", (*fields.values(), run_id))


def get_run(run_id: str) -> dict | None:
    with conn() as c:
        return c.execute("select * from runs where id = %s", (run_id,)).fetchone()


def log_step(run_id: str, node: str, status: str, output=None, error: str | None = None, ms: int = 0):
    with conn() as c:
        c.execute(
            "insert into steps (run_id, node, status, output, error, ms) values (%s, %s, %s, %s, %s, %s)",
            (run_id, node, status, Jsonb(output), error, ms),
        )


def get_steps(run_id: str) -> list[dict]:
    with conn() as c:
        return c.execute("select node, status, output, error, ms, created_at from steps where run_id = %s order by id", (run_id,)).fetchall()


def list_runs(status: str | None = None) -> list[dict]:
    with conn() as c:
        q = "select id, goal, status, updated_at from runs" + (" where status = %s" if status else "") + " order by created_at"
        return c.execute(q, (status,) if status else ()).fetchall()


def save_studies(rows: list[dict], note: str) -> int:
    """Record approved studies (rows in Spotter's shape, see library.rows) locally."""
    with conn() as c:
        c.cursor().executemany(
            """insert into library (pmid, title, journal, year, types, url, note)
               values (%(pmid)s, %(title)s, %(journal)s, %(year)s, %(types)s, %(url)s, %(note)s)
               on conflict (pmid) do update set note = excluded.note""",
            [{**r, "note": note} for r in rows],
        )
        return len(rows)
