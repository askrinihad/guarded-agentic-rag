"""
Phase 2 - RAGAS Evaluation
Runs every question in the golden dataset through the running API,
collects answers + retrieved chunks, then scores them with RAGAS.

PREREQUISITE: the API must be running first in another terminal:
    uvicorn src.api.main:app --reload

Run this script from the project root:
    python eval/ragas_eval.py
"""

import json
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/ask"
GOLDEN_DATASET_PATH = Path("eval/golden_dataset.json")
RESULTS_PATH = Path("eval/results.json")


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_questions_through_api(golden_dataset):
    """Send each question to the running API and collect the response."""
    records = []
    for item in golden_dataset:
        print(f"Asking [{item['id']}]: {item['question']}")
        response = requests.post(API_URL, json={"question": item["question"]})
        response.raise_for_status()
        result = response.json()

        records.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["retrieved_chunks"],
                "ground_truth": item["ground_truth"],
                "difficulty": item["difficulty"],
                "id": item["id"],
            }
        )
    return records


def run_ragas_evaluation(records):
    """
    Scores records with RAGAS. This section is the most likely to need
    adjustment - RAGAS's API for wrapping a local Hugging Face model as
    the judge LLM has changed across versions, and I can't guarantee
    this matches what's installed. If this raises an ImportError or
    TypeError, paste the traceback and we'll fix it together.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from transformers import pipeline as hf_pipeline

    # Reuse the same local generation model as the RAGAS judge.
    # Note: a 1.5B model is a weak judge compared to GPT-4-class models
    # RAGAS was originally benchmarked with - treat these scores as
    # directionally useful, not absolute ground truth.
    judge_pipe = hf_pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=512,
    )
    judge_llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=judge_pipe))

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    return results


def main():
    if RESULTS_PATH.exists():
        print(f"Found existing {RESULTS_PATH} - skipping API calls, reusing saved answers.")
        print("(Delete this file first if you want to re-ask the questions, e.g. after code changes.)\n")
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        golden_dataset = load_golden_dataset()
        print(f"Loaded {len(golden_dataset)} questions from golden dataset\n")

        records = run_questions_through_api(golden_dataset)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\nSaved raw answers + retrieved chunks to {RESULTS_PATH}")

    print("\nRunning RAGAS evaluation (this will take a while on CPU)...")
    ragas_results = run_ragas_evaluation(records)

    print("\n=== RAGAS Scores ===")
    print(ragas_results)

if __name__ == "__main__":
    main()