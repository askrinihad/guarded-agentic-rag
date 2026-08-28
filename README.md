# Guarded Agentic RAG

A retrieval-augmented question-answering system with agentic tool use, systematic evaluation, and tested defenses against prompt injection. Built over a research paper corpus, running entirely on local infrastructure (Qdrant + a local LLM, no external API dependency).

## Results

| Metric | Baseline | After fix | Delta |
|---|---|---|---|
| Faithfulness | 0.850 | 1.000 | +0.150 |
| Answer relevancy | 0.671 | 0.681 | +0.010 |
| Context precision | 0.452 | 0.460 | +0.008 |
| Context recall | 0.548 | 0.596 | +0.048 |

Root cause: fixed-size chunking split facts mid-sentence and flattened a results table into unlabeled numbers. Fix: sentence-aware chunking, top-4 → top-6 retrieval. Full detail: [`eval/`](eval/).

## Architecture

```
                     ┌─────────────┐
   query ──────────▶ │   FastAPI    │
                     │   endpoint   │
                     └──────┬─────┘
                            │
                     ┌──────▼──────┐
                     │  LangGraph   │
                     │    agent     │
                     └──┬───────┬──┘
                        │       │
              ┌─────────▼─┐   ┌─▼───────────┐
              │  Retrieval │   │  arXiv       │
              │  tool      │   │  search tool │
              │  (Qdrant)  │   │              │
              └───────────┘   └──────────────┘
```

PDF → sentence-aware chunking → embedding (Qdrant) → cosine-similarity retrieval → grounded generation. The agent layer adds ReAct-style multi-tool reasoning on top of the same retrieval core.

## Engineering findings

**Chunking strategy is a first-order lever on RAG quality**, not a minor detail — see Results above.

**Automatic prompt optimization is not a free win on small models.** DSPy's `BootstrapFewShot`, benchmarked against a hand-written prompt on a held-out test set, underperformed it (0.224 vs. 0.416) and produced one confirmed hallucination. Documented in [`prompt_opt/`](prompt_opt/).

**Native LangGraph tool-calling assumes a capability small local models don't reliably have.** `create_react_agent` silently skipped tool use and fabricated a citation. Replaced with a manual ReAct loop (`StateGraph`) that reliably routes between retrieval and arXiv search. See [`src/agent/`](src/agent/).

**Prompt injection (OWASP LLM01) is a confirmed retrieval-layer risk.** A poisoned document, planted in the live collection, won 1–2 of the top-6 retrieval slots against the real paper across repeated tests — pure semantic relevance, no special placement. Generation-layer hijack was attempted with two attack variants and did not succeed on this model; the untrusted-content-tagging mitigation is applied regardless, since retrieval-layer compromise is real independent of generation-layer outcome. See [`security/`](security/).

## Tech stack

| Layer | Choice |
|---|---|
| Vector store | Qdrant |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Generation | Qwen2.5-1.5B-Instruct (local) |
| Agent orchestration | LangGraph |
| Tool protocol | MCP (Model Context Protocol) |
| Prompt optimization | DSPy |
| Evaluation | Custom RAGAS-style pipeline (LLM-as-judge + embedding metrics) |
| Serving | FastAPI |
| Infra | Docker |

## Quick start

```bash
git clone https://github.com/askrinihad/guarded-agentic-rag.git && cd guarded-agentic-rag
python3 -m venv ragEnv && source ragEnv/bin/activate
pip install -r requirements.txt

docker run -d --name qdrant-guarded-rag -p 6333:6333 -p 6334:6334 \\
  -v qdrant_storage:/qdrant/storage qdrant/qdrant

mkdir -p data/papers && cp /path/to/your.pdf data/papers/
python src/ingestion/ingest.py
uvicorn src.api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \\
  -d '{"question": "your question"}'
```

Evaluation, agent, and security tests: `eval/`, `src/agent/agent.py`, `security/injection_test.py`.

## Project structure

```
guarded-agentic-rag/
|-- src/
|   |-- ingestion/       # PDF chunking + embedding
|   |-- agent/           # LangGraph ReAct agent
|   `-- api/             # FastAPI service
|-- eval/                # golden dataset + RAGAS-style scoring
|-- prompt_opt/          # DSPy optimization experiment
|-- security/            # prompt injection attack/defense test
`-- data/papers/          # source PDFs (gitignored)
```
