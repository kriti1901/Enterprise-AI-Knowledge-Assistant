"""Retrieval and answer-quality evaluation over a golden question set.

Golden file format (JSONL), one object per line:
  {"question": "...", "relevant_title": "<document title>", "expected": "<keyword in a correct answer>"}

Metrics:
  hit_at_k          share of questions whose relevant document appears in the top-k chunks
  mrr               mean reciprocal rank of the first relevant chunk
  answer_accuracy   share of agent answers containing the expected keyword (case-insensitive)
  avg_latency_ms    mean end-to-end agent latency
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.config import get_settings
from app.db import Document, init_db, new_session
from app.llm import get_llm
from app.retrieval import search
from app.services import ingest_document


def default_eval_dir() -> Path:
    return Path(get_settings().eval_dir)


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_eval(
    session: Session,
    departments: list[str],
    golden_path: Path | None = None,
    k: int | None = None,
    with_answers: bool = True,
) -> dict[str, Any]:
    k = k or get_settings().top_k
    golden = load_golden(golden_path or default_eval_dir() / "golden_set.jsonl")
    llm = get_llm() if with_answers else None

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    correct = 0
    latencies: list[float] = []
    per_question = []

    for row in golden:
        started = time.perf_counter()
        hits = search(row["question"], departments, session, mode="hybrid", top_k=k)
        rank = next((i for i, h in enumerate(hits, start=1) if h.title == row["relevant_title"]), None)
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        answer_ok = None
        if llm is not None:
            result = run_agent(row["question"], departments, session, llm)
            answer_ok = row.get("expected", "").lower() in result.answer.lower()
            correct += int(answer_ok)
        latencies.append((time.perf_counter() - started) * 1000)
        per_question.append(
            {"question": row["question"], "rank": rank, "answer_ok": answer_ok}
        )

    n = len(golden) or 1
    return {
        "num_questions": len(golden),
        "k": k,
        "hit_at_k": round(hits_at_k / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "answer_accuracy": round(correct / n, 4) if llm is not None else None,
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "details": per_question,
    }


def seed_sample_corpus(session: Session, root: Path) -> int:
    """Ingest eval/sample_docs/<Department>/*.txt, skipping titles that already exist."""
    added = 0
    for path in sorted(root.glob("*/*.txt")):
        department = path.parent.name
        title = path.stem.replace("_", " ")
        exists = session.scalar(select(Document.id).where(Document.title == title))
        if exists:
            continue
        ingest_document(
            session,
            filename=path.name,
            content=path.read_bytes(),
            title=title,
            department=department,
            uploaded_by="eval-seed",
        )
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval and answer evaluation")
    parser.add_argument("--seed", action="store_true", help="ingest the sample corpus first")
    parser.add_argument("--departments", default="", help="comma-separated departments to evaluate as")
    parser.add_argument("--no-answers", action="store_true", help="skip LLM answer generation")
    args = parser.parse_args()

    init_db()
    session = new_session()
    eval_dir = default_eval_dir()
    if args.seed:
        added = seed_sample_corpus(session, eval_dir / "sample_docs")
        print(f"Seeded {added} sample documents")
    departments = [d.strip() for d in args.departments.split(",") if d.strip()]
    if not departments:
        departments = sorted({d.name for d in (eval_dir / "sample_docs").iterdir() if d.is_dir()})
    metrics = run_eval(session, departments, with_answers=not args.no_answers)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
