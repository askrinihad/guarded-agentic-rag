"""
Guarded Agentic RAG - hosted deployment variant

Same app as deploy/app.py, with ONE change: embeddings come from the
Hugging Face Inference API instead of running sentence-transformers
locally. This removes torch (~400MB) from the deployment, fitting
within Render's 512MB free tier.

Critically, it calls the SAME model (all-MiniLM-L6-v2, 384 dimensions)
used during ingestion - so the existing Qdrant collection stays valid
and vectors remain comparable. Using a different embedding model here
would silently break retrieval.

deploy/app.py (local embeddings) is kept as the reference implementation
and matches the pipeline the README's evaluation numbers were measured on.

Environment variables required:
    QDRANT_URL
    QDRANT_API_KEY
    GROQ_API_KEY
    HF_TOKEN        (Hugging Face access token, read scope is enough)
"""

import os

import gradio as gr
import requests
from groq import Groq
from qdrant_client import QdrantClient

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBED_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 6

QDRANT_URL = os.environ["QDRANT_URL"].strip()
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"].strip()
GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
HF_TOKEN = os.environ["HF_TOKEN"].strip()

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def embed(text: str) -> list[float]:
    """Get a 384-dim embedding from the HF Inference API.

    Uses the feature-extraction pipeline, which returns the sentence
    embedding for this model. Raises on API errors so failures surface
    clearly rather than silently returning bad vectors.
    """
    response = requests.post(
        HF_EMBED_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    response.raise_for_status()
    vector = response.json()

    # The feature-extraction endpoint may return either a flat vector or a
    # nested list depending on input shape - normalise to a flat list.
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]

    if not isinstance(vector, list) or len(vector) != 384:
        raise ValueError(f"Unexpected embedding shape from HF API: {type(vector)}")

    return vector


def retrieve(question: str, top_k: int = TOP_K) -> list[str]:
    query_vector = embed(question)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return [point.payload["text"] for point in results]


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks))
    return (
        "Answer the question using only the excerpts below. "
        "If the answer isn't in the excerpts, say you don't know.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_question(question: str, show_chunks: bool):
    if not question.strip():
        return "Please enter a question.", ""

    try:
        chunks = retrieve(question)
    except Exception as e:
        return f"Retrieval failed: {e}", ""

    if not chunks:
        return "No relevant content found in the paper.", ""

    prompt = build_prompt(question, chunks)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Generation failed: {e}", ""

    chunks_display = ""
    if show_chunks:
        chunks_display = "\n\n---\n\n".join(
            f"**Excerpt {i+1}**\n\n{c}" for i, c in enumerate(chunks)
        )

    return answer, chunks_display


with gr.Blocks(title="Guarded Agentic RAG") as demo:
    gr.Markdown(
        "# Guarded Agentic RAG\n"
        "Ask a question about *A Knowledge-Driven Incremental Learning Framework "
        "for Automating and Enhancing Plant Identification in Airports* "
        "(IEEE CASE 2025).\n\n"
        "This system answers **only** from that paper - questions outside its "
        "content will return 'I don't know'. That grounding is deliberate and is "
        "what the evaluation in the "
        "[GitHub repo](https://github.com/askrinihad/guarded-agentic-rag) measures.\n\n"
        "*Note: the live demo uses hosted inference (Groq) for speed; the "
        "benchmarked evaluation numbers in the repo were measured on a local "
        "model pipeline. First request may be slow while the free instance wakes up.*"
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Your question",
            placeholder="e.g. What architecture does the paper use for feature extraction?",
            scale=4,
        )
        submit_btn = gr.Button("Ask", scale=1, variant="primary")

    show_chunks_toggle = gr.Checkbox(
        label="Show retrieved excerpts (see what the answer was grounded on)",
        value=False,
    )

    answer_output = gr.Textbox(label="Answer", lines=8)
    chunks_output = gr.Markdown()

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, show_chunks_toggle],
        outputs=[answer_output, chunks_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, show_chunks_toggle],
        outputs=[answer_output, chunks_output],
    )

    gr.Examples(
        examples=[
            "What continual learning method does the paper use?",
            "What accuracy did the proposed model achieve on the Oxford 102 dataset?",
            "What robotic platform was used to collect real-world images?",
            "How does the framework avoid catastrophic forgetting?",
        ],
        inputs=question_input,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
