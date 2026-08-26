# Guarded Agentic RAG

A retrieval-augmented agent over a research paper corpus, built to demonstrate RAG evaluation, prompt optimization, agentic orchestration, and prompt-injection defense — the four core skills for GenAI/Agentic AI engineering roles.

## Why this project

Most RAG demos stop at "it answers questions." This one measures whether it answers *correctly* (RAGAS-style eval), optimizes the prompt systematically instead of by hand (DSPy), extends into a tool-using agent (LangGraph + MCP), and demonstrates a real prompt-injection attack and its mitigation (OWASP LLM01).

## Status

🚧 In progress — Phases 1-2 complete, see roadmap below.

## Roadmap

- [x] **Phase 1 — RAG core**: Chunked + embedded the IEEE CASE 2025 paper into Qdrant. FastAPI endpoint for retrieval + generation (local Qwen2.5-1.5B-Instruct).
- [x] **Phase 2 — Evaluation**: 20-question hand-verified golden dataset. Baseline eval exposed two real failure modes (mid-sentence fact truncation, flattened tables). Fixed with sentence-aware chunking + wider top-k retrieval. Faithfulness improved 0.85 → 1.00. See Results below.
- [ ] **Phase 3 — Prompt optimization**: DSPy-optimized prompt vs. hand-written baseline, evaluated on the same golden dataset.
- [ ] **Phase 4 — Agentic layer**: LangGraph agent with a retrieval tool + a second tool (arXiv search / citation formatter), exposed as MCP servers where possible.
- [ ] **Phase 5 — Prompt injection defense**: Planted indirect injection in a retrieved document. Demonstrated attack, then one mitigation (untrusted-content tagging / output filtering), with before/after write-up.

## Architecture
                 ┌─────────────┐

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

## Project structure
