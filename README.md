# Agent Orchestrator

A research agent for lifter bloodwork and supplement questions, and the offline, human-approved builder of [Spotter](https://spotterlabs.vercel.app)'s shared study library: give it a goal ("Does creatine harm kidney function?"), and it plans, calls tools, reflects, and answers with PubMed citations. Writes to shared data wait for a human.

**Stack:** Python · LangGraph · Claude API · Redis · Postgres · FastAPI · Celery

```
POST /runs ──> FastAPI ──> Celery worker ──> LangGraph:  plan ──> act ──> tools ──> reflect ─┐
                                                          ^        ^                         │
                                                          │        └──── next step ──────────┤
                                                          └───────── re-plan ────────────────┘ done ──> answer
     state saved to Redis after every node (checkpointer)        every node logged to Postgres
```

| Piece | Where |
|---|---|
| **1. Graph** with plan, act and reflect nodes. `act` only chooses a tool; a separate `tools` node runs it, so resuming after approval doesn't call the model again | `orchestrator/graph.py` |
| **2. Tools** each registered with a typed Pydantic input schema and a risk level. Arguments are validated before anything runs; bad input comes back to the model as an observation | `orchestrator/tools.py` |
| **3. Memory** is LangGraph state, checkpointed to Redis (`RedisSaver`) after every node. Kill the worker mid-run and the next one picks up where it stopped | `orchestrator/worker.py` |
| **4. Human in the loop:** high-risk tools (`save_to_library`) call `interrupt()`. The run waits as `waiting_approval` until `POST /runs/{id}/approve` | `orchestrator/graph.py`, `orchestrator/api.py` |
| **5. Step log:** every node writes node, status, output, error and timing to Postgres. `/history` lists checkpoints; `/replay` forks from any of them, or resumes a failed run | `orchestrator/db.py`, `orchestrator/api.py` |

Tools: `search_pubmed`, `read_abstracts` (NCBI E-utilities), `fetch_page` (Scrapling; public URLs only, pages through long text, jumps to a keyword), `save_to_library` (high risk; writes to Spotter's study library).

## How it feeds Spotter

Spotter's live Ask tab stays on its own fast path (one Vercel function, Groq, a few seconds). This service runs offline and makes the library Ask starts from deeper:

```
seeds.txt ──> research run ──> save_to_library ──> waits for you ──> approve ──> Spotter's public.studies
(common       plan / search /    only PMIDs the      (CLI or          re-fetched     via the add_studies RPC,
 questions)   read / reflect     agent read          /approve)        from PubMed    same rows Ask writes
                                                                                         │
            Spotter Ask: finds title matches in the library, reads the abstracts, cites them
```

- **Same shape as Spotter.** Approved PMIDs are re-fetched from PubMed and written as `{pmid, title, journal, year, types, url}` through Spotter's `add_studies(rows jsonb)` RPC (`supabase/ask.sql` in the Spotter repo), so they look exactly like studies Ask cited itself. No schema change. Spotter has no column for the reviewer's note, so the note and every approval are kept here, in the local `library` table (or `library.jsonl` with `cli run`).
- **Only what it read.** `save_to_library` drops PMIDs the agent never read with `read_abstracts`, and refuses outright (without asking you) if it read none of them. The approval prompt shows each study's title and link.
- **No false "saved".** In testing the model once wrote "Saved to Spotter's library" without calling the tool. Seed runs now pass `require: ["save_to_library"]`, and a run can't finish until that tool was actually attempted (your rejection counts).
- **Local by default.** `LIBRARY_TARGET=local` (the default) never leaves your machine. Writing to Spotter needs `LIBRARY_TARGET=spotter` plus `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

On Spotter's side, `findStudies` in `src/ask/store.ts` looks up title matches and hands up to five to the model as a starting point; it still has to read an abstract before it can cite it.

The model is Claude (`claude-opus-5-5`, adaptive thinking, `medium` effort, with the server-side refusal fallback). Set `LLM_PROVIDER=groq` to use Groq's free API instead.

## Run it

Free: set `LLM_PROVIDER=groq` with a free Groq key. The worker runs one research run at a time so Groq's free tier (8k tokens a minute) isn't rate limited by its own runs.

```bash
cp .env.example .env    # set ANTHROPIC_API_KEY (or LLM_PROVIDER=groq + GROQ_API_KEY)
docker compose up --build
docker compose exec api python -m orchestrator.cli seed     # one run per question in seeds.txt
docker compose exec api python -m orchestrator.cli review --wait   # approve or reject each save as it comes in
```

No Docker? Run the whole thing in one process (in-memory state, approvals asked inline, approved saves logged to `library.jsonl`):

```bash
pip install -r requirements.txt
LLM_PROVIDER=groq GROQ_API_KEY=... python -m orchestrator.cli run seeds.txt
```

To write approved studies into Spotter itself, add `LIBRARY_TARGET=spotter`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to `.env`. Spotter's `supabase/ask.sql` has to have been run on that project first.

Or locally, with Redis 8+ and Postgres running:

```bash
pip install -r requirements.txt
celery -A orchestrator.worker worker --loglevel=info &
uvicorn orchestrator.api:app
```

## API

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"goal": "Does creatine harm kidney function in healthy adults? Save the best 2 studies to the library.", "require": ["save_to_library"]}'
# {"id": "…", "status": "queued"}

curl localhost:8000/runs?status=waiting_approval   # what needs a decision
curl localhost:8000/runs/ID            # status: queued | running | waiting_approval | done | failed
                                        # + pending_approval {tool, input, studies} while waiting, answer when done
curl -X POST localhost:8000/runs/ID/approve -H 'content-type: application/json' -d '{"approved": false, "reason": "only meta-analyses"}'
curl localhost:8000/runs/ID/steps      # step log: node, status, output, error, ms
curl localhost:8000/runs/ID/history    # checkpoints, newest first
curl -X POST localhost:8000/runs/ID/replay -H 'content-type: application/json' -d '{"checkpoint_id": "…"}'  # fork from there
curl -X POST localhost:8000/runs/ID/replay -H 'content-type: application/json' -d '{}'                      # resume a failed run
```

A rejected action goes back to the agent as an observation ("A human rejected this action…"), and it carries on without it. Runs stop after 8 reflect cycles and answer with what they have.

## Tests

```bash
python -m pytest orchestrator   # graph, approval/rejection, iteration cap, schemas, SSRF guard, parsing,
                                # the Spotter write path (row shape, RPC call, local-only default), CLI approval
```

The tests use a scripted fake model, an in-memory checkpointer and faked PubMed and Supabase, so they need no keys, Redis or Postgres, and can't touch Spotter's production library.
