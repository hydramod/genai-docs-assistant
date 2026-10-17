# docs-assistant

Assignment: build a small agentic workflow. It's an assistant that answers questions about **Fernhill Works**, a made-up coworking space, using only its member handbook (`docs/`), and says so honestly when the handbook doesn't cover something. It uses RAG with Chroma, a LangGraph agent with a calculator tool, Langfuse tracing and an LLM-judged eval.

| File | What it does |
| --- | --- |
| `docs/` | The handbook: 5 Markdown files, 25 paragraphs, written for this project |
| `ingest.py` | Step 1: `chunk_text` splits the docs into overlapping chunks, which are stored in a local Chroma database (`chroma_db/`) |
| `llm.py` | Shared plumbing: `.env` loading, the Claude client, retry with backoff, refusal handling, and Langfuse cost reporting |
| `assistant.py` | Step 2: `answer_question` retrieves the top chunks and asks Claude to answer from them only |
| `agent.py` | Step 3: a LangGraph agent with two tools, `search_handbook` and a safe `calculator`, that decides which to use |
| `evaluate.py` + `eval_set.json` | Step 5: 12 questions with notes on what a good answer contains, a `judge` that scores 1-5, and a report |
| `test_*.py` | 36 pytest tests, including "the chunker produces overlapping chunks of the expected size" |
| `eval_results/` | Saved eval runs (see Results below) |

## Setup

1. Get an Anthropic API key from https://platform.claude.com.
2. Optional, for Step 4: create a free project at https://cloud.langfuse.com and copy its public and secret keys.
3. Copy `.env.example` to `.env` and paste your keys in. `.env` is gitignored.

```
uv run python ingest.py       # builds chroma_db/ (first run downloads an ~80 MB embedding model)
uv run pytest -v              # no API key needed
```

## Run it

```
uv run python assistant.py                                  # demo questions, incl. one it can't answer
uv run python assistant.py "Can I bring my dog on a Monday?"
uv run python agent.py                                      # demo: retrieval-only, tool-only, and both
uv run python agent.py "What would 3 extra hours in Cedar cost?"
```

## Observability (Step 4)

With Langfuse keys in `.env`, every question becomes a trace in the Langfuse dashboard:

- `answer_question` → `retrieve` (a retriever span listing the chunks found) → `generate` → `claude` (a generation with model, token counts and cost)
- `run_agent` → `claude` → `search_handbook` / `calculator` tool spans → `claude` ...

Each span shows its timing. Cost is calculated from the token counts at $4 / $20 per million input / output tokens for Claude Opus 5.5, with thinking tokens counted as output because that's how they're billed. Langfuse 5 would also record the Anthropic SDK's own spans automatically, which shows every call twice and doubles the cost. `llm.py` filters those out. Without keys, tracing switches itself off and everything else still works.

## Evaluation (Step 5)

```
uv run python evaluate.py --top-k 2 --label baseline
uv run python evaluate.py --top-k 5 --label more-chunks --baseline eval_results/baseline.json
```

The first command scores a baseline. The second is the "one improvement": retrieving more chunks per question. It compares against the baseline and lists which questions went up or down. Other improvements to try are re-chunking (`uv run python ingest.py --chunk-size 300 --overlap 60`, then re-run the eval), or editing `SYSTEM_PROMPT` in `assistant.py`. Add `--agent` to evaluate the LangGraph agent instead of plain RAG.

Two eval questions are deliberately not in the handbook (a gym, the Wi-Fi password). A good answer to those says the handbook doesn't cover it instead of inventing something.

### Results (Claude Opus 5.5, 12 questions)

| Run | Chunks per question | Average score |
| --- | --- | --- |
| `baseline-original-judge.json` | 2 | 3.67 (judge flaw, see below) |
| `baseline.json` | 2 | 4.08 |
| `more-chunks.json` | 5 | **4.67** (+0.58) |

**What the first run taught us about the judge.** At first the judge saw only the short notes, so it docked points for true facts the notes didn't mention, like the 10% annual discount. The fix was to give the judge the whole handbook so it can check every claim. The baseline was then re-run with the fixed judge, so both runs above are scored the same way. Comparing runs scored by different judges would not have been fair.

**What the improvement fixed.** With 2 chunks, three answers missed facts that were in the handbook. The assistant honestly said "the handbook doesn't say" rather than inventing anything, but that was still wrong. Retrieving 5 chunks raised all three to 5/5, including the fridge question (1 → 5).

**What's still wrong.** "Can I come in on a Sunday with a Flex membership?" scores 1 even with 5 chunks. The word "Flex membership" pulls every retrieved chunk towards membership prices, so the "closed on Sundays" chunk never makes the top 5. The agent gets it right: it searches a second time for "opening hours weekend". Plain RAG would need a better retriever, such as hybrid keyword + vector search or rewriting the question before searching.

## Design notes

- **Embeddings run locally.** Chroma's default model (all-MiniLM-L6-v2) needs no API key.
- **Model:** Claude Opus 5.5 at low effort for answers and medium effort for judging. Change it with `CLAUDE_MODEL` / `CLAUDE_EFFORT` in `.env`.
- **Refusal fallbacks are on** (`fallbacks: "default"`). If Claude's safety filter declines a harmless question by mistake, the API reruns it on a fallback model instead of failing.
- **Retries:** `with_retry` retries rate limits, overloads and network errors with exponential backoff and jitter. It never retries errors that won't fix themselves, like a bad request or a bad key. The SDK's built-in retries are switched off so the two don't stack.
- **The calculator never uses `eval()`.** It parses the expression and allows only numbers and arithmetic, so model output can't run code.
- **The judge uses structured output**, so its reply is always valid JSON with a score from 1 to 5. It gives its reason before its score.
