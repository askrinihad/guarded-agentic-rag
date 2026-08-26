"""
Phase 2 - Custom Evaluation (RAGAS-style, without the ragas package)

The installed `ragas` package has a broken import chain on this machine
(references langchain_community.chat_models.vertexai, which doesn't exist
in the installed langchain_community version, and reinstalling didn't fix
it). Rather than keep fighting version conflicts, this script computes the
same core ideas directly using tools already working in this project:

- Faithfulness: asks the local Qwen model to judge whether the answer is
  actually supported by the retrieved context (LLM-as-judge, same idea
  RAGAS uses, just without RAGAS's wrapper machinery).
- Answer relevancy: cosine similarity between the question and answer
  embeddings (using the same all-MiniLM-L6-v2 model from ingestion).
- Context precision: average similarity between the question and each
  retrieved chunk - are the retrieved chunks actually about the question?
- Context recall: similarity between the ground truth answer and the best-
  matching retrieved chunk - did retrieval find the right information?

These are simplified compared to official RAGAS (which uses more elaborate
prompting and claim-decomposition), but they measure the same things and
are honestly labelled as a custom implementation, not RAGAS's own numbers.

Run from the project root (results.json must already exist from ragas_eval.py):
    python eval/custom_eval.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline

RESULTS_PATH = Path("eval/results.json")
SCORED_PATH = Path("eval/results_scored.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_records():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faithfulness_prompt(answer, contexts):
    context_text = "\n\n".join(contexts)
    return (
        "You are evaluating whether an AI-generated answer is faithful to "
        "the source material it was given.\n\n"
        f"SOURCE EXCERPTS:\n{context_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        "Is every claim in the answer directly supported by the source "
        "excerpts above? Respond with exactly one word - YES or NO - "
        "followed by a one-sentence reason.\n"
        "Response:"
    )


def score_faithfulness(judge, answer, contexts):
    if not contexts or not answer.strip():
        return 0.0, "No context or empty answer"

    prompt = build_faithfulness_prompt(answer, contexts)
    output = judge(prompt, max_new_tokens=60, do_sample=False)
    generated = output[0]["generated_text"][len(prompt):].strip()

    score = 1.0 if generated.upper().startswith("YES") else 0.0
    return score, generated


def score_answer_relevancy(embedder, question, answer):
    if not answer.strip():
        return 0.0
    q_emb = embedder.encode(question, convert_to_tensor=True)
    a_emb = embedder.encode(answer, convert_to_tensor=True)
    return float(util.cos_sim(q_emb, a_emb).item())


def score_context_precision(embedder, question, contexts):
    if not contexts:
        return 0.0
    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_embs = embedder.encode(contexts, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, c_embs)[0]
    return float(sims.mean().item())


def score_context_recall(embedder, ground_truth, contexts):
    if not contexts:
        return 0.0
    gt_emb = embedder.encode(ground_truth, convert_to_tensor=True)
    c_embs = embedder.encode(contexts, convert_to_tensor=True)
    sims = util.cos_sim(gt_emb, c_embs)[0]
    return float(sims.max().item())  # best matching chunk


def main():
    records = load_records()
    print(f"Loaded {len(records)} records from {RESULTS_PATH}\n")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading judge model ({JUDGE_MODEL})... this may take a moment.")
    judge = hf_pipeline("text-generation", model=JUDGE_MODEL, max_new_tokens=60)

    scored = []
    for r in records:
        print(f"Scoring [{r['id']}]...")

        faithfulness, judge_reason = score_faithfulness(judge, r["answer"], r["contexts"])
        relevancy = score_answer_relevancy(embedder, r["question"], r["answer"])
        precision = score_context_precision(embedder, r["question"], r["contexts"])
        recall = score_context_recall(embedder, r["ground_truth"], r["contexts"])

        scored.append(
            {
                **r,
                "scores": {
                    "faithfulness": faithfulness,
                    "faithfulness_reason": judge_reason,
                    "answer_relevancy": round(relevancy, 3),
                    "context_precision": round(precision, 3),
                    "context_recall": round(recall, 3),
                },
            }
        )

    with open(SCORED_PATH, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"\nSaved scored results to {SCORED_PATH}")

    # Summary
    n = len(scored)
    avg_faith = sum(s["scores"]["faithfulness"] for s in scored) / n
    avg_rel = sum(s["scores"]["answer_relevancy"] for s in scored) / n
    avg_prec = sum(s["scores"]["context_precision"] for s in scored) / n
    avg_rec = sum(s["scores"]["context_recall"] for s in scored) / n

    print("\n=== Average Scores ===")
    print(f"Faithfulness:      {avg_faith:.3f}")
    print(f"Answer Relevancy:  {avg_rel:.3f}")
    print(f"Context Precision: {avg_prec:.3f}")
    print(f"Context Recall:    {avg_rec:.3f}")

    print("\n=== By Difficulty ===")
    for level in ["easy", "medium", "hard"]:
        subset = [s for s in scored if s["difficulty"] == level]
        if not subset:
            continue
        avg = sum(s["scores"]["faithfulness"] for s in subset) / len(subset)
        print(f"{level}: {len(subset)} questions, avg faithfulness = {avg:.3f}")


if __name__ == "__main__":
    main()