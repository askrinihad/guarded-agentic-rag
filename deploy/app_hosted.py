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
from groq import Groq
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 6
MIN_SCORE = 0.15  # below this, retrieval found nothing genuinely relevant

# Fixed facts about the paper, always available to the model.
#
# These are handled outside retrieval deliberately. Dense embedding search
# maps queries by topical gist, and questions like "who wrote this paper"
# are topically about authorship - which doesn't match content that is
# topically about plant identification, even when the author names are
# literally present in it. Measured: "who wrote the paper" returned a top
# similarity score of 0.05, i.e. no meaningful match anywhere in the corpus.
# Fixed metadata belongs in the prompt, not the vector store.
PAPER_METADATA = (
    "Paper: 'A Knowledge-Driven Incremental Learning Framework for Automating "
    "and Enhancing Plant Identification in Airports'\n"
    "Authors: Nihad Askri, Ferhat Attal, Abdelghani Chibani, Karim Djouani, "
    "Ilies Chibane, Reda Belaiche, Yacine Amirat\n"
    "Affiliation: Univ Paris Est Creteil, LISSI, F-94400 Vitry, France\n"
    "Venue: IEEE CASE 2025\n"
    "Funding: OLGA H2020 European project"
)

QDRANT_URL = os.environ["QDRANT_URL"].strip()
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"].strip()
GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
HF_TOKEN = os.environ["HF_TOKEN"].strip()

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(token=HF_TOKEN)


def embed(text: str) -> list[float]:
    """Get a 384-dim embedding via the official HF client.

    Uses InferenceClient rather than a hardcoded URL - Hugging Face has
    migrated their inference infrastructure, and the client handles
    endpoint routing internally rather than depending on a URL that
    may change.
    """
    result = hf_client.feature_extraction(text, model=EMBEDDING_MODEL)

    vector = result.tolist() if hasattr(result, "tolist") else list(result)

    # Normalise nested output to a flat 384-dim vector
    if vector and isinstance(vector[0], list):
        vector = vector[0]

    if len(vector) != 384:
        raise ValueError(f"Unexpected embedding dimension: {len(vector)}")

    return vector


def retrieve(question: str, top_k: int = TOP_K) -> list[str]:
    """Retrieve relevant chunks, discarding matches below MIN_SCORE.

    Without the threshold, a query with no good match still returns the
    six least-bad chunks - which then get presented to the model as if
    they were relevant context. Filtering lets the system distinguish
    'the paper does not cover this' from 'here is loosely related text'.
    """
    query_vector = embed(question)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return [p.payload["text"] for p in results if p.score >= MIN_SCORE]


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks))
    return (
        "Answer the question using only the paper metadata and excerpts "
        "below. If the answer isn't there, say you don't know.\n\n"
        f"PAPER METADATA:\n{PAPER_METADATA}\n\n"
        f"EXCERPTS FROM THE PAPER:\n{context}\n\n"
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
        # Retrieval found nothing above the relevance threshold - but the
        # metadata may still answer the question, so try generation anyway
        # with metadata only rather than refusing outright.
        prompt = build_prompt(question, [])
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            answer = (completion.choices[0].message.content or "").strip()
            return answer or "I don't have information about that in this paper.", ""
        except Exception as e:
            return f"Generation failed: {e}", ""

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

    # Markdown, not Textbox - the model returns Markdown formatting
    # (**bold**, numbered lists), which a Textbox would show as raw asterisks.
    answer_output = gr.Markdown(label="Answer")
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
