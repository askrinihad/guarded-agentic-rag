# Guarded Agentic RAG

A retrieval-augmented, tool-using question-answering system built over a research paper - designed to demonstrate not just that a RAG pipeline can be built, but that it can be rigorously evaluated, systematically improved, and tested against real security risks.

Ask it a question about the source paper and it retrieves relevant excerpts, grounds its answer in them, and can extend into agentic tool use (searching arXiv for related work) when the question calls for it. Every claim about quality in this README is backed by a measured before/after comparison, not a guess.

## What this demonstrates

- **RAG done properly**: chunking, embedding, vector retrieval, and grounded generation - plus the evaluation harness to actually know if it's working
- **Evaluation-driven iteration**: a hand-built golden dataset and a RAGAS-style scoring pipeline caught two real retrieval bugs, which were fixed and the fix was measured (faithfulness 0.85 to 1.00)
- **Honest evaluation of automatic prompt optimization**: tested DSPy against a hand-written prompt and reported a genuine negative result, including a caught hallucination - not everything that's supposed to help, helps
- **Agentic tool use from first principles**: built a LangGraph agent, found the standard prebuilt approach fails silently on small local models, and built a working manual ReAct loop instead
- **Security testing**: planted and tested indirect prompt injection attacks against the live retrieval pipeline, confirmed a real retrieval-layer vulnerability, and applied a standard mitigation

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

A PDF is chunked (sentence-aware, not fixed-character) and embedded into Qdrant. Questions are embedded the same way, retrieved by cosine similarity, and answered by a local LLM grounded in the retrieved excerpts. The agent layer adds tool-based reasoning on top of the same retrieval core.

## Key findings

Each of these came from actually testing the system, not from assumption - full detail and raw data linked below.

**Chunking strategy materially affects answer quality.** A baseline evaluation on a 20-question golden dataset found 3 failures traced to two root causes: fixed-size chunking cutting facts mid-sentence, and a results table getting flattened into an unlabeled wall of numbers. Switching to sentence-aware chunking and widening retrieval from top-4 to top-6 chunks raised faithfulness from 0.850 to 1.000 and context recall from 0.548 to 0.596. See [`eval/`](eval/).

**Automatic prompt optimization needs care on small models.** DSPy's BootstrapFewShot optimizer was tested against the hand-written baseline prompt on a held-out test set. It scored lower (0.224 vs. 0.416 on a word-overlap metric), and inspection of the actual outputs found why: one empty generation failure, and one confirmed hallucination - the optimized prompt invented future-work content (satellite imagery, multi-language support) that appears nowhere in the source paper. See [`prompt_opt/`](prompt_opt/).

**Native LangGraph tool-calling silently failed on a small local model.** The standard create_react_agent approach never called a tool at all - it skipped straight to a hallucinated, fabricated citation. A manual ReAct loop (explicit text-based tool-call parsing instead of relying on native function-calling) was built and hardened through several real failure modes until it worked reliably end-to-end. See [`src/agent/`](src/agent/).

**Prompt injection is a real, confirmed retrieval-layer risk.** A poisoned document, planted in the same live collection as the real paper, won 1-2 of the top 6 retrieval slots for a relevant question across two separate test runs - outranking legitimate content purely on semantic similarity. Generation-layer hijacking was attempted with two distinct attack styles and did not succeed on this specific model/prompt setup, an honest and specific finding, not a general claim that injection doesn't work. The standard untrusted-content-tagging mitigation is applied regardless, since the retrieval-layer compromise is real. See [`security/`](security/).

## Tech stack

- **Retrieval**: Qdrant (vector database), sentence-transformers (all-MiniLM-L6-v2)
- **Generation**: Qwen2.5-1.5B-Instruct, run locally via Hugging Face transformers
- **Agentic orchestration**: LangGraph (StateGraph, manual ReAct loop)
- **Prompt optimization**: DSPy
- **Evaluation**: custom RAGAS-style metrics (LLM-as-judge faithfulness, embedding-similarity relevancy/precision/recall) - the ragas package itself had a broken dependency chain in this environment; see eval/ragas_eval.py for details
- **Serving**: FastAPI
- **Infrastructure**: Docker (Qdrant container)

## Rebuild this yourself

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Docker Desktop, installed and running
- A PDF you want to build a Q&A system over

### 1. Clone and set up the environment

```bash
git clone https://github.com/askrinihad/guarded-agentic-rag.git
cd guarded-agentic-rag

python3 -m venv ragEnv
source ragEnv/bin/activate        # Windows: ragEnv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install transformers torch accelerate langchain_huggingface requests arxiv
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant-guarded-rag \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify: open http://localhost:6333/dashboard in a browser, or run `docker ps` and confirm the container is Up.

### 3. Add your document

```bash
mkdir -p data/papers
cp /path/to/your/paper.pdf data/papers/
```

### 4. Ingest and embed

```bash
python src/ingestion/ingest.py
```

This chunks the PDF (sentence-aware), embeds each chunk, and stores it in a fresh Qdrant collection.

### 5. Run the API

```bash
uvicorn src.api.main:app --reload
```

Test it at http://127.0.0.1:8000/docs, or:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

### 6. (Optional) Run the evaluation suite

Build your own golden dataset following the format in eval/golden_dataset.json, then:

```bash
python eval/ragas_eval.py
python eval/custom_eval.py
```

### 7. (Optional) Run the agent

```bash
python src/agent/agent.py
```

### 8. (Optional) Run the prompt injection test

```bash
python security/injection_test.py
```

This temporarily plants a poisoned document into the live collection, tests it, then automatically cleans up afterward.

## Project structure

```
guarded-agentic-rag/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   `-- papers/              # source PDFs (gitignored)
|-- src/
|   |-- ingestion/            # chunking + embedding into Qdrant
|   |-- agent/                 # LangGraph ReAct agent
|   `-- api/                   # FastAPI app
|-- eval/
|   |-- golden_dataset.json
|   |-- ragas_eval.py          # collects answers; RAGAS scoring notes inside
|   |-- custom_eval.py         # working custom RAGAS-style scorer
|   `-- results_scored.json
|-- prompt_opt/
|   |-- dspy_optimize.py
|   `-- results.json
`-- security/
    `-- injection_test.py
```

## License

MIT
