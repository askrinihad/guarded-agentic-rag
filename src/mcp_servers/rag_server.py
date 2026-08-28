"""
MCP server exposing this project's two tools over the Model Context Protocol,
so any MCP-compatible client (Claude Desktop, Cursor, other agents) can use
them - not just the local ReAct agent in src/agent/agent.py.

Same underlying logic as the agent's tools; the difference is the interface:
plain Python functions become spec-compliant MCP tools via FastMCP decorators.

Install:
    pip install "mcp[cli]"

Run directly (stdio transport, for local MCP clients):
    python src/mcp_servers/rag_server.py

Environment variables (optional - falls back to local Docker Qdrant):
    QDRANT_URL       e.g. https://xxxx.cloud.qdrant.io:6333
    QDRANT_API_KEY   your cluster's database API key

To connect from Claude Desktop, add to its MCP config:
    {
      "mcpServers": {
        "guarded-rag": {
          "command": "python",
          "args": ["/absolute/path/to/src/mcp_servers/rag_server.py"]
        }
      }
    }
"""

import os

from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 6

QDRANT_URL = (os.environ.get("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.environ.get("QDRANT_API_KEY") or "").strip()

mcp = MCPServer("guarded-rag")

# Loaded once at import; MCP clients keep the server process alive between calls
embedder = SentenceTransformer(EMBEDDING_MODEL)

if QDRANT_URL and QDRANT_API_KEY:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    # Fall back to the local Docker instance used during development
    qdrant = QdrantClient(host="localhost", port=6333)


@mcp.tool()
def retrieve_paper_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Search the ingested research paper for passages relevant to a question.

    The corpus is 'A Knowledge-Driven Incremental Learning Framework for
    Automating and Enhancing Plant Identification in Airports' (IEEE CASE
    2025). Use this for any question about that paper's methods, results,
    datasets, or architecture. Returns the most semantically similar
    excerpts, separated by blank lines.

    Args:
        question: What to search for. Use a topical query, not a paper title.
        top_k: How many excerpts to return (default 6).
    """
    query_vector = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    if not results:
        return "No relevant content found in the paper."
    return "\n\n".join(point.payload["text"] for point in results)


@mcp.tool()
def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for papers related to a topic.

    Use this to find related work or citations beyond the ingested paper.
    Returns a list of titles, publication years, and arXiv URLs.

    Args:
        query: Topic to search for, e.g. 'continual learning image classification'.
        max_results: How many papers to return (default 3).
    """
    try:
        import arxiv
    except ImportError:
        return "arxiv package not installed. Run: pip install arxiv"

    import time
    time.sleep(3)  # respect arXiv's rate limit

    try:
        client = arxiv.Client(delay_seconds=3, num_retries=2)
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        results = [
            f"- {r.title} ({r.published.year}): {r.entry_id}"
            for r in client.results(search)
        ]
        if not results:
            return "No related papers found on arXiv."
        return "\n".join(results)
    except Exception as e:
        return f"arXiv search failed (possibly rate-limited): {e}"


if __name__ == "__main__":
    mcp.run()
