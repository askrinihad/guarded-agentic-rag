"""
Phase 4 - Agentic layer (v2 - manual ReAct loop)

The first attempt used LangGraph's prebuilt create_react_agent, which
expects the LLM to support structured tool-calling. Qwen2.5-1.5B-Instruct
didn't reliably do that - it skipped tool use entirely and hallucinated an
answer. This version builds a manual ReAct loop using LangGraph's
StateGraph directly: we prompt the model to state its intent in a simple
text format we parse ourselves, then route to the right tool node.

Run from the project root:
    python src/agent/agent.py
"""

import re
from typing import TypedDict

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from langgraph.graph import StateGraph, END

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 6
MAX_STEPS = 4

embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(host="localhost", port=6333)
generator = hf_pipeline("text-generation", model=GENERATION_MODEL, max_new_tokens=300, do_sample=False)


def retrieve_paper_context(query: str) -> str:
    query_vector = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=TOP_K
    ).points
    if not results:
        return "No relevant content found in the paper."
    return "\n\n".join(point.payload["text"] for point in results)


def search_arxiv(query: str) -> str:
    try:
        import arxiv
    except ImportError:
        return "arxiv package not installed. Run: pip install arxiv"

    import time
    time.sleep(3)  # be polite to arXiv's rate limit, especially across repeated runs

    try:
        client = arxiv.Client(delay_seconds=3, num_retries=2)
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for result in client.results(search):
            results.append(f"- {result.title} ({result.published.year}): {result.entry_id}")
        if not results:
            return "No related papers found on arXiv."
        return "\n".join(results)
    except Exception as e:
        return f"arXiv search failed (likely rate-limited): {e}"


TOOLS = {
    "retrieve_paper_context": retrieve_paper_context,
    "search_arxiv": search_arxiv,
}


class AgentState(TypedDict):
    question: str
    scratchpad: str
    steps: int
    answer: str


REACT_PROMPT_TEMPLATE = """You are an assistant that can use tools to answer questions.

Available tools:
- retrieve_paper_context[query]: search the ingested research paper. Use a query about the PAPER'S TOPIC, never a paper title you found elsewhere.
- search_arxiv[query]: search arXiv for related papers

To use a tool, respond with exactly one line in this format:
ACTION: tool_name[query text]

When you have enough information to answer, respond with exactly:
ACTION: finish[your final answer here]

RULES:
- If the question asks about the paper's own methods, call retrieve_paper_context
  at least once before finishing.
- Once you have called retrieve_paper_context AND search_arxiv at least once each,
  you have enough information - call finish on your next turn. Do not repeat
  a tool call you have already made.

THE ORIGINAL QUESTION (do not lose track of this): {question}

What you have learned so far:
{scratchpad}

Respond with exactly one ACTION line, nothing else. Base your query on the
ORIGINAL QUESTION above, not on titles or facts from the OBSERVATIONS."""


def reason_node(state: AgentState) -> AgentState:
    prompt = REACT_PROMPT_TEMPLATE.format(question=state["question"], scratchpad=state["scratchpad"])
    output = generator(prompt, max_new_tokens=150, do_sample=False)
    generated = output[0]["generated_text"][len(prompt):].strip()

    # Accept the tool call with or without a literal "ACTION:" prefix, and
    # restrict tool names to the known set - this model sometimes drops the
    # "ACTION:" label even when it correctly picks a real tool call format.
    # Accept either [ ] or ( ) around the argument - this small model
    # isn't perfectly consistent about which bracket style it uses.
    match = re.search(
        r"(?:ACTION:\s*)?(retrieve_paper_context|search_arxiv|finish)[\[\(](.*?)[\]\)]",
        generated,
        re.DOTALL,
    )
    if not match:
        state["answer"] = generated
        state["scratchpad"] += "\n[Model did not follow ACTION format, using raw output as answer]"
        return state

    tool_name, tool_arg = match.group(1), match.group(2).strip()
    state["scratchpad"] += f"\nACTION: {tool_name}[{tool_arg}]"

    if tool_name == "finish":
        state["answer"] = tool_arg
    elif tool_name in TOOLS:
        observation = TOOLS[tool_name](tool_arg)
        state["scratchpad"] += f"\nOBSERVATION: {observation[:500]}"
    else:
        state["scratchpad"] += f"\nOBSERVATION: Unknown tool '{tool_name}'."

    state["steps"] += 1
    return state


def give_up_node(state: AgentState) -> AgentState:
    # A proper node, not a conditional-edge function - only node return
    # values persist back into graph state. Mutating state inside
    # should_continue() below does NOT stick, which was the earlier bug.
    state["answer"] = "Could not reach an answer within the step limit."
    return state


def should_continue(state: AgentState) -> str:
    if state.get("answer"):
        return "end"
    if state["steps"] >= MAX_STEPS:
        return "give_up"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("give_up", give_up_node)
    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason", should_continue, {"continue": "reason", "give_up": "give_up", "end": END}
    )
    graph.add_edge("give_up", END)
    return graph.compile()


def main():
    app = build_graph()

    test_question = (
        "What continual learning method does the paper use, and can you find "
        "a related paper on arXiv about continual learning for image classification?"
    )
    print(f"Asking: {test_question}\n")

    result = app.invoke({"question": test_question, "scratchpad": "", "steps": 0, "answer": ""})

    print("=== Full trace ===")
    print(result["scratchpad"])
    print("\n=== Final answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
