"""Evaluate local NumPy vector retrieval against the frozen holdout set."""

import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from notebooks.evaluation_utils import evaluate
from notebooks.vector_search import OpenAIEmbedder, VectorSearchIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("vector benchmark blocked: OPENAI_API_KEY is not configured")
    documents = json.loads(args.corpus.read_text(encoding="utf-8"))
    with args.ground_truth.open(newline="", encoding="utf-8") as source:
        holdout = [row for row in csv.DictReader(source) if row["split"] == "holdout"]
    index = VectorSearchIndex(documents, OpenAIEmbedder(OpenAI(), args.model))
    metrics = evaluate(lambda query, k: index.search(query, k), holdout, k=args.k)
    result = {
        "backend": "vector",
        "model": args.model,
        "split": "holdout",
        **metrics,
    }
    output = args.ground_truth.with_name("vector_evaluation_result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Vector Hit Rate@{args.k}={metrics[f'hit_rate_at_{args.k}']:.4f}, MRR={metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
