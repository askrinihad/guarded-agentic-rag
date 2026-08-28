"""
Guarded Agentic RAG - live demo
Retrieval: Qdrant Cloud (same sentence-aware-chunked collection as the local project)
Generation: Groq (Llama 3.1 8B) - fast hosted inference, chosen for demo responsiveness.
            Note: the evaluation numbers in the main README were measured with the
            local Qwen2.5-1.5B-Instruct pipeline in this repo, not this demo model.
            This demo prioritizes response speed for visitors; the repo's eval/
            folder documents the actual measured quality of the local pipeline.
"""

import os

import gradio as gr
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ["QDRANT_URL"].strip()
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"].strip()
GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 6

embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using only the excerpts below. "
        "If the answer isn't in the excerpts, say you don't know.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_question(question: str):
    if not question.strip():
        return "Please enter a question.", ""

    query_vector = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=TOP_K
    ).points
    context_chunks = [point.payload["text"] for point in results]

    prompt = build_prompt(question, context_chunks)

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )
    answer = completion.choices[0].message.content

    chunks_display = "\n\n---\n\n".join(
        f"**Excerpt {i+1}**\n\n{c}" for i, c in enumerate(context_chunks)
    )

    return answer, chunks_display


with gr.Blocks(title="Guarded Agentic RAG") as demo:
    gr.Markdown(
        "# Guarded Agentic RAG - Live Demo\n"
        "Ask a question about the ingested research paper "
        "(*A Knowledge-Driven Incremental Learning Framework for Automating "
        "and Enhancing Plant Identification in Airports*, IEEE CASE 2025).\n\n"
        "This demo uses Qdrant Cloud for retrieval and Groq (Llama 3.1 8B) for fast "
        "generation. The full evaluation results in the "
        "[GitHub repo](https://github.com/askrinihad/guarded-agentic-rag) were "
        "measured with a different local model pipeline - see the repo's README "
        "for the actual benchmarked numbers."
    )

    question_input = gr.Textbox(
        label="Your question",
        placeholder="e.g. What continual learning method does the paper use?",
    )
    ask_button = gr.Button("Ask", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=5)

    with gr.Accordion("Show retrieved chunks (see what the model was grounded on)", open=False):
        chunks_output = gr.Markdown()

    ask_button.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, chunks_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, chunks_output],
    )

    gr.Examples(
        examples=[
            "What continual learning method does the paper use?",
            "What accuracy did the proposed model achieve on the Oxford 102 dataset?",
            "What robotic platform was used to collect real-world images?",
        ],
        inputs=question_input,
    )

if __name__ == "__main__":
    # Render (and most PaaS hosts) assign a port via the PORT env var and
    # require binding to 0.0.0.0, not localhost, to be reachable externally.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
