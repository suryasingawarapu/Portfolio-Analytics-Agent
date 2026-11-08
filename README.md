# Portfolio Analytics Agent

An agent that answers natural-language questions about a portfolio database — either by generating SQL or by calculating sector exposure directly — powered by Gemini. Usable both as a command-line tool and through an optional chat UI.

## Accuracy

**10/10** on the provided ground truth questions, run via `evaluate.py`:

| Difficulty | Score |
|---|---|
| Easy | 3/3 |
| Medium | 5/5 |
| Hard | 2/2 |
| **Total** | **10/10** |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt

cp .env.example .env                # add your GEMINI_API_KEY
```

`portfolio.db` ships pre-built, so there's no setup step beyond that.

### Run via command line

```bash
python main.py                                       # interactive chat, in the terminal
python main.py "How many portfolios do we have?"      # one-off question, prints the answer and exits
python evaluate.py                                    # runs all 10 ground truth questions, prints the accuracy table above
```

### Run via Streamlit (optional chat UI)

```bash
streamlit run streamlit_app.py
```

Opens a chatbot-style interface in your browser. For every answer, you get:
- A **natural-language reply**, streamed in word-by-word as it's generated (not just dumped on screen all at once)
- A **"Show the data behind this answer"** panel underneath, containing:
  - The **exact SQL query** that was generated and run (or, for sector exposure, the SQL used to pull the underlying holdings)
  - A **chart** — a bar chart for sector exposure or any multi-row numeric result, or a metric scorecard for a single aggregate number (like a COUNT or SUM)
  - The full data as a table

So you can read the plain-English answer at a glance, then expand to verify exactly how it was computed.

## How it works

| File | Role |
|---|---|
| `agent.py` | Routes each question to a tool, runs it, returns the answer |
| `sql_tool.py` | Question → SQL (via Gemini) → validated → executed read-only |
| `exposure.py` | Sector exposure, calculated directly in Python (no LLM) |
| `example_bank.py` | Few-shot examples retrieved to help Gemini on harder questions |
| `main.py` / `streamlit_app.py` | Command-line and optional chat interfaces |
| `respond.py` | Streamlit-only: phrases the result as a streamed natural-language reply |
| `evaluate.py` | Runs the ground truth set, reports accuracy |
| `config.py` | API setup, retries, logging |

**Routing:** a clearly-worded sector exposure question skips the AI entirely and goes straight to the Python calculator (free, instant, exact). Everything else goes through Gemini, which picks a tool, runs it, and can chain a second tool if the question needs it (e.g. "sector exposure of the portfolio with the highest AUM" looks up the portfolio first, then its exposure).

## Key design choices

- **Dates normalized on load** — source files mixed `DD-MM-YYYY` and `YYYY-MM-DD` formats inconsistently; everything in `portfolio.db` is `YYYY-MM-DD`.
- **Sector exposure excludes bonds and renormalizes** to 100% of equity value — bonds have no sector, and including them in the denominator would understate every sector.
- **Portfolio names are never guessed** — exact match, then a fuzzy match, but only when it resolves to exactly one candidate.
- **Generated SQL is treated as untrusted** — validated as a single read-only query, and executed on a connection that's read-only at the database level too.
- **Evaluation compares values, not exact SQL or column sets** — two different queries can both be correct; only the actual data matters.
- **Repeated questions are cached in memory** (not on disk) — instant on a repeat ask, and failures are never cached so a transient error doesn't stick around.

## What's not included, and why

- **No RAG / vector DB / GraphRAG over the dataset** — the whole schema is a few hundred rows; it fits in a prompt, no retrieval system needed. (Few-shot *example* retrieval for SQL generation is the one place this pattern is actually used, at a scale where it's justified.)
- **No LangChain/LangGraph** — the routing logic is a short loop; a framework wouldn't add capability here.
- **No Langfuse / tracing platform** — plain logging already shows what the agent is doing, which is what was asked for.
- **No FastAPI / hosted API** — the deliverable is a CLI agent.

## Known limitation

Gemini's free tier caps requests **per day, per project** (as low as 20/day on some projects), not just per minute. A long evaluation run can hit this — it's an account limit, not a bug.
