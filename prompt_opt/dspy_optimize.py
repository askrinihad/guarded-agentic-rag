"""
Phase 3 - DSPy Prompt Optimization

Compares the hand-written prompt in src/api/main.py against a DSPy-optimized
prompt, on the same golden dataset, split into train (used by DSPy's
optimizer) and test (held out, used only for the final comparison).

UNCERTAINTY FLAG: DSPy's API for wrapping a *local* Hugging Face model
(as opposed to a hosted API) has changed across versions. The LocalQwenLM
class below is a best-effort implementation for DSPy 3.x's BaseLM
interface - if this raises an error on import or on first call, that's
expected as a starting point to debug from, not a sign something is
fundamentally wrong with the approach.

Run from the project root:
    python prompt_opt/dspy_optimize.py
"""
from types import SimpleNamespace
import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

import dspy

GOLDEN_DATASET_PATH = Path("eval/golden_dataset.json")
COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 6

TEST_IDS = {"q08", "q09", "q10", "q15", "q20"}


class LocalQwenLM(dspy.BaseLM):
    """Wraps the local Qwen pipeline so DSPy can call it like any other LM.
    This is the part most likely to need adjustment for your installed
    DSPy version - if it errors, paste the traceback and we'll fix it."""

    def __init__(self, model_name=GENERATION_MODEL, max_new_tokens=300):
        super().__init__(model=model_name)
        self.pipe = hf_pipeline("text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def forward(self, prompt=None, messages=None, **kwargs):
        if messages:
            prompt = "\n".join(m.get("content", "") for m in messages)

        output = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0]["generated_text"][len(prompt):].strip()

        message = SimpleNamespace(role="assistant", content=generated)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return SimpleNamespace(
            choices=[choice],
            model=self.model,
            id="local-qwen",
            created=0,
            object="chat.completion",
            usage=usage,
        )


class AnswerFromContext(dspy.Signature):
    """Answer the question using only the provided context. If the answer
    isn't in the context, say you don't know."""

    context: str = dspy.InputField(desc="retrieved excerpts from the source document")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(AnswerFromContext)

    def forward(self, context, question):
        return self.generate(context=context, question=question)


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_context(embedder, qdrant, question, top_k=TOP_K):
    query_vector = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return "\n\n".join(point.payload["text"] for point in results)


def build_examples(golden_dataset, embedder, qdrant):
    """Pre-retrieves context for every question once, so DSPy's optimizer
    isn't re-querying Qdrant on every trial."""
    examples = []
    for item in golden_dataset:
        context = retrieve_context(embedder, qdrant, item["question"])
        example = dspy.Example(
            context=context,
            question=item["question"],
            answer=item["ground_truth"],
        ).with_inputs("context", "question")
        example.id = item["id"]
        examples.append(example)
    return examples


def faithfulness_metric(example, pred, trace=None):
    """Simple metric for DSPy's optimizer: reward answers whose words
    substantially overlap with the ground truth. Crude compared to the
    LLM-judge faithfulness in custom_eval.py, but fast enough to run
    across many optimization trials."""
    gt_words = set(example.answer.lower().split())
    pred_words = set(pred.answer.lower().split())
    if not gt_words:
        return 0.0
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap


def main():
    print("Loading embedder and connecting to Qdrant...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    qdrant = QdrantClient(host="localhost", port=6333)

    print("Loading local Qwen model for DSPy...")
    lm = LocalQwenLM()
    dspy.configure(lm=lm)

    golden_dataset = load_golden_dataset()
    train_items = [d for d in golden_dataset if d["id"] not in TEST_IDS]
    test_items = [d for d in golden_dataset if d["id"] in TEST_IDS]
    print(f"Train: {len(train_items)} questions, Test: {len(test_items)} questions")

    print("Retrieving context for all questions (once)...")
    train_examples = build_examples(train_items, embedder, qdrant)
    test_examples = build_examples(test_items, embedder, qdrant)

    baseline_module = RAGModule()

    print("\nRunning baseline (unoptimized DSPy prompt) on test set...")
    baseline_results = []
    for ex in test_examples:
        pred = baseline_module(context=ex.context, question=ex.question)
        score = faithfulness_metric(ex, pred)
        baseline_results.append({"id": ex.id, "question": ex.question, "answer": pred.answer, "score": score})
        print(f"  [{ex.id}] score={score:.2f}")

    print("\nOptimizing prompt with DSPy (this will take a while on CPU)...")
    optimizer = dspy.BootstrapFewShot(metric=faithfulness_metric, max_bootstrapped_demos=4)
    optimized_module = optimizer.compile(RAGModule(), trainset=train_examples)

    print("\nRunning optimized DSPy prompt on the same test set...")
    optimized_results = []
    for ex in test_examples:
        pred = optimized_module(context=ex.context, question=ex.question)
        score = faithfulness_metric(ex, pred)
        optimized_results.append({"id": ex.id, "question": ex.question, "answer": pred.answer, "score": score})
        print(f"  [{ex.id}] score={score:.2f}")

    baseline_avg = sum(r["score"] for r in baseline_results) / len(baseline_results)
    optimized_avg = sum(r["score"] for r in optimized_results) / len(optimized_results)

    print("\n=== Results ===")
    print(f"Baseline (hand-written-style prompt) avg overlap score: {baseline_avg:.3f}")
    print(f"DSPy-optimized prompt avg overlap score:                {optimized_avg:.3f}")

    with open("prompt_opt/results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"baseline": baseline_results, "optimized": optimized_results},
            f, indent=2, ensure_ascii=False,
        )
    print("\nSaved detailed results to prompt_opt/results.json")


if __name__ == "__main__":
    main()
