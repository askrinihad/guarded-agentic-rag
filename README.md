# Guarded Agentic RAG

A retrieval-augmented agent over a research paper corpus, built to demonstrate RAG evaluation, prompt optimization, agentic orchestration, and prompt-injection defense — the four core skills for GenAI/Agentic AI engineering roles.

## Why this project

Most RAG demos stop at "it answers questions." This one measures whether it answers *correctly* (RAGAS-style eval), optimizes the prompt systematically instead of by hand (DSPy), extends into a tool-using agent (LangGraph + MCP), and demonstrates a real prompt-injection attack and its mitigation (OWASP LLM01).

## Status

🚧 In progress — Phases 1-4 complete, see roadmap below.

## Roadmap

- [x] **Phase 1 — RAG core**: Chunked + embedded the IEEE CASE 2025 paper into Qdrant. FastAPI endpoint for retrieval + generation (local Qwen2.5-1.5B-Instruct).
- [x] **Phase 2 — Evaluation**: 20-question hand-verified golden dataset. Baseline eval exposed two real failure modes (mid-sentence fact truncation, flattened tables). Fixed with sentence-aware chunking + wider top-k retrieval. Faithfulness improved 0.85 → 1.00. See Results below.
- [x] **Phase 3 — Prompt optimization**: DSPy-optimized prompt vs. hand-written baseline, evaluated on a 5-question held-out test set. Result: DSPy underperformed the hand-written prompt (see Results below) — a genuine, documented negative finding.
- [x] **Phase 4 — Agentic layer**: LangGraph agent with retrieval + arXiv search tools. First attempt (native tool-calling via create_react_agent) failed silently on the small local model; a manual ReAct loop succeeded after several debugging rounds. See Results below.
- [ ] **Phase 5 — Prompt injection defense**: Planted indirect injection in a retrieved document. Demonstrated attack, then one mitigation (untrusted-content tagging / output filtering), with before/after write-up.

## Architecture

```
                     ┌─────────────┐
   query ──────────▶ │   FastAPI    │
                     │   endpoint   │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  LangGraph   │
                     │    agent     │
                     └──┬───────┬──┘
                        │       │
              ┌─────────▼─┐   ┌─▼───────────┐
              │  Retrieval │   │  arXiv /     │
              │  tool      │   │  citation    │
              │  (Qdrant)  │   │  tool (MCP)  │
              └────────────┘   └──────────────┘
```

## Tech stack

- **Retrieval**: Qdrant, sentence-transformers (all-MiniLM-L6-v2)
- **Generation**: Qwen2.5-1.5B-Instruct (local, via Hugging Face transformers)
- **Orchestration**: LangGraph (Phase 4)
- **Tool protocol**: MCP (Phase 4)
- **Evaluation**: custom RAGAS-style metrics (faithfulness via LLM-as-judge, relevancy/precision/recall via embedding similarity) — the `ragas` package itself had a broken dependency chain in this environment, documented in `eval/ragas_eval.py`
- **Prompt optimization**: DSPy (Phase 3)
- **Serving**: FastAPI, Docker

## Setup

```bash
python3 -m venv ragEnv
source ragEnv/bin/activate  # Windows: ragEnv\Scripts\activate
pip install -r requirements.txt
pip install transformers torch accelerate
```

Requires Docker running locally for Qdrant:
```bash
docker run -d --name qdrant-guarded-rag -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Results

**Phase 2 evaluation** — 20-question golden dataset, hand-verified against the source paper, scored with a custom RAGAS-style pipeline (`eval/custom_eval.py`).

| Metric | Baseline (fixed-size chunks, top-4) | After fix (sentence-aware chunks, top-6) | Change |
|---|---|---|---|
| Faithfulness | 0.850 | **1.000** | +0.150 |
| Answer Relevancy | 0.671 | 0.681 | +0.010 |
| Context Precision | 0.452 | 0.460 | +0.008 |
| Context Recall | 0.548 | 0.596 | +0.048 |

**What the baseline run found**: 3 of 20 questions scored 0 on faithfulness, all "easy" factual-lookup questions asking for specific numbers (embedding dimensionality, an accuracy percentage, GPU memory size). Manual inspection of the retrieved chunks showed two distinct root causes:
1. Fixed 500-character chunking cut sentences mid-fact, so the correct information either landed in a chunk that didn't make the top-4 retrieval cut, or was truncated within its own chunk.
2. A results table in the paper got flattened into a wall of numbers with no row/column labels during chunking, causing the model to pick a plausible but wrong value from the wrong column.

**The fix**: rewrote chunking to group whole sentences instead of slicing at raw character offsets (`src/ingestion/ingest.py`), and increased retrieval from top-4 to top-6 chunks (`src/api/main.py`). Re-running the same 20 questions against the rebuilt index eliminated all three faithfulness failures.

**Known remaining gap**: the table-flattening problem is only partially addressed by wider retrieval — a proper fix would need structure-aware extraction for tables specifically, not just better sentence splitting. Candidate for a future pass.



## Phase 3 finding: DSPy underperformed the hand-written prompt

Tested whether DSPy's `BootstrapFewShot` optimizer could beat the hand-written
prompt in `src/api/main.py`, using a 15-question train / 5-question held-out
test split of the golden dataset, with `Qwen2.5-1.5B-Instruct` as both the
generator and the model being optimized.

| | Baseline (hand-written prompt) | DSPy-optimized prompt |
|---|---|---|
| Avg. word-overlap score (5 test questions) | 0.416 | 0.224 |

DSPy scored lower, but the raw number understates what's actually going on -
inspecting the five individual answers showed a mixed picture:

- **2 of 5** DSPy answers were factually correct but more concise than the
  reference wording, and got penalized by the (crude) word-overlap metric
  for it - not a real quality problem, a metric artifact.
- **1 of 5** came back as an empty string - a genuine generation failure,
  likely related to the added few-shot examples increasing prompt length
  on a small 1.5B model.
- **1 of 5** was a confirmed hallucination: the optimized prompt produced an
  answer about "satellite imagery," "remote sensing," and "multiple
  languages" for a future-work question - none of which appear anywhere in
  the source paper. The actual future work section discusses expanding to
  new species and ontology-based reasoning, not remote sensing or language
  support.

**Conclusion**: few-shot prompt optimization, applied without adjustment to
a small local model, traded groundedness for pattern-matching against the
optimizer's own scoring signal - it optimized toward matching reference
*wording*, not toward staying faithful to the source document. The
hand-written direct-instruction prompt was more reliable for this model
size, even though it wasn't automatically tuned. This is a genuine,
documented negative result, not a failed experiment - worth knowing before
reaching for automatic prompt optimization as a default, particularly on
small/local models with limited context-following capacity.

Full raw outputs: `prompt_opt/results.json`.



## Phase 4 finding: native tool-calling failed, manual ReAct loop succeeded

Built an agent with two tools (search the ingested paper; search arXiv for
related work) using `Qwen2.5-1.5B-Instruct` as the reasoning model.

**First attempt** used LangGraph's prebuilt `create_react_agent`, which
expects the LLM to natively support structured tool-calling. The model
never called a tool at all - it skipped straight to a hallucinated answer,
inventing a nonexistent paper ("Continual Learning with Deep Networks" by
Yann LeCun, 2015 - this paper does not exist).

**Fix**: rewrote the agent as a manual ReAct loop using LangGraph's
`StateGraph` directly - prompting the model to state its intent in a
simple parseable text format (`ACTION: tool_name[query]`) instead of
relying on native tool-calling. This is more robust for small models but
needed several rounds of hardening against real failure modes encountered
along the way:
- the model dropping the literal word "ACTION:" from its output
- the model switching between `[...]` and `(...)` bracket styles
- the model querying `retrieve_paper_context` with an arXiv paper *title*
  it had just read, instead of a query about its own paper's topic
- a state-mutation bug where a fallback message set inside a LangGraph
  conditional-edge function silently failed to persist (only node return
  values update graph state, not router-function side effects)

**Final result**, after prompting the model to keep the original question
in view and hardening the parser against format drift: the agent correctly
called both tools in sequence, retrieved its own paper's real method
(the similarity-based knowledge base, not a hallucination), searched
arXiv, and terminated cleanly via `finish`.

**Known remaining limitation**: the final answer under-uses what it
retrieved - it confirms "continual learning is used" without stating the
specific mechanism, even though that detail was present in its own tool
observation. A larger model would likely synthesize this; the 1.5B model
here reliably executes the tool-use loop but doesn't reliably reason over
what the tools returned.

Full run traces documented in commit history for `src/agent/agent.py`.

## Project structure
