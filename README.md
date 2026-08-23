# Guarded Agentic RAG

A retrieval-augmented agent over a research paper corpus, built to demonstrate RAG evaluation, prompt optimization, agentic orchestration, and prompt-injection defense.

## Status
In progress.

## Roadmap
- [ ] Phase 1 — RAG core: chunk + embed papers into Qdrant, FastAPI endpoint
- [ ] Phase 2 — Evaluation: golden Q&A dataset + RAGAS metrics
- [ ] Phase 3 — Prompt optimization: DSPy vs hand-written baseline
- [ ] Phase 4 — Agentic layer: LangGraph agent + MCP tools
- [ ] Phase 5 — Prompt injection defense: attack + mitigation writeup

## Tech stack
Qdrant, FastAPI, LangGraph, MCP, RAGAS, DSPy, Docker

## Project structure
guarded-agentic-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── data/papers/
├── src/{ingestion,retrieval,agent,mcp_servers,api}/
├── eval/
├── prompt_opt/
├── security/
└── docs/

## License
MIT
