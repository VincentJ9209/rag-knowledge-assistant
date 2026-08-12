"""Reproducible keyword retrieval evaluation on the frozen FAQ corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

from sqlitesearch import TextSearchIndex

from notebooks.evaluation_utils import evaluate


BASELINE = (2.0, 0.5)
CANDIDATES = (
    BASELINE,
    (1.0, 0.5),
    (1.5, 0.5),
    (2.5, 0.5),
    (3.0, 0.5),
    (2.0, 0.0),
    (2.0, 1.0),
    (2.5, 1.0),
)


def build_keyword_index(documents: list[dict[str, str]], db_path: Path) -> TextSearchIndex:
    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=str(db_path),
    )
    if index.count() != 0:
        raise ValueError("evaluation index must start empty")
    for document in documents:
        index.add(document)
    return index


def keyword_search(
    index: TextSearchIndex,
    query: str,
    k: int,
    *,
    question_boost: float,
    section_boost: float,
) -> list[dict[str, str]]:
    return index.search(
        query=query,
        boost_dict={"question": question_boost, "section": section_boost},
        filter_dict={"course": "llm-zoomcamp"},
        num_results=k,
    )


def _measure(index, records, weights, split):
    question_boost, section_boost = weights
    metrics = evaluate(
        lambda query, k: keyword_search(
            index,
            query,
            k,
            question_boost=question_boost,
            section_boost=section_boost,
        ),
        records,
        k=5,
    )
    return {
        "backend": "keyword",
        "variant": "baseline" if weights == BASELINE else "candidate",
        "split": split,
        "question_boost": question_boost,
        "section_boost": section_boost,
        "hit_rate_at_5": metrics["hit_rate_at_5"],
        "mrr": metrics["mrr"],
        "question_count": metrics["question_count"],
    }


def run_evaluation(documents, ground_truth):
    tuning = [row for row in ground_truth if row["split"] == "tuning"]
    holdout = [row for row in ground_truth if row["split"] == "holdout"]
    with tempfile.TemporaryDirectory(prefix="rag-keyword-evaluation-") as temp_dir:
        index = build_keyword_index(documents, Path(temp_dir) / "evaluation.db")
        try:
            tuning_rows = [_measure(index, tuning, weights, "tuning") for weights in CANDIDATES]
            selected = max(
                tuning_rows,
                key=lambda row: (row["hit_rate_at_5"], row["mrr"]),
            )
            selected_weights = (selected["question_boost"], selected["section_boost"])
            baseline_holdout = _measure(index, holdout, BASELINE, "holdout")
            candidate_holdout = _measure(index, holdout, selected_weights, "holdout")
        finally:
            index.close()
    candidate_holdout["variant"] = "selected_candidate"
    return tuning_rows + [baseline_holdout, candidate_holdout]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = json.loads(args.corpus.read_text(encoding="utf-8"))
    with args.ground_truth.open(newline="", encoding="utf-8") as source:
        ground_truth = list(csv.DictReader(source))
    rows = run_evaluation(documents, ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["split"] == "holdout":
            print(
                f"{row['variant']}: Hit Rate@5={row['hit_rate_at_5']:.4f}, "
                f"MRR={row['mrr']:.4f}, weights={row['question_boost']}/{row['section_boost']}"
            )


if __name__ == "__main__":
    main()
