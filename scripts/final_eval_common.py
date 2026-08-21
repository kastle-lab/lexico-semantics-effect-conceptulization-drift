#!/usr/bin/env python3
"""Shared paths and canonical RDF tuple utilities for final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES_DIR = ROOT / "evaluation_dependencies"
HUMAN_REVIEW_DEPENDENCIES_DIR = DEPENDENCIES_DIR / "human_review"
ARTIFACTS_PATH = DEPENDENCIES_DIR / "frozen_query_artifacts.csv"
REQUIREMENTS_PATH = DEPENDENCIES_DIR / "cq_requirements.jsonl"
SCHEMA_ELEMENTS_PATH = DEPENDENCIES_DIR / "kg_schema_elements.csv"
HUMAN_REVIEW_SELECTION_PATH = (
    HUMAN_REVIEW_DEPENDENCIES_DIR / "human_review_selection.csv"
)
HUMAN_QUERY_REVIEW_PATH = (
    HUMAN_REVIEW_DEPENDENCIES_DIR / "human_query_review.csv"
)
REFERENCE_DIR = ROOT / "analysis/reference_answers"
REFERENCE_PATH = REFERENCE_DIR / "reference_answer_sets.jsonl"
RESULTS_PATH = REFERENCE_DIR / "artifact_result_sets.jsonl"
CAPTURE_INDEX_PATH = REFERENCE_DIR / "artifact_result_capture_index.csv"
EXECUTION_DIR = ROOT / "analysis/execution"
EXECUTION_METRICS_PATH = EXECUTION_DIR / "artifact_execution_metrics.csv"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def canonical_term(term: Any) -> tuple[Any, ...]:
    if term is None:
        return ("unbound",)
    if not isinstance(term, dict):
        raise ValueError("Answer terms must be typed JSON objects or null")
    term_type = term.get("type")
    value = term.get("value")
    if term_type not in {"uri", "literal", "bnode"} or not isinstance(
        value, str
    ):
        raise ValueError(f"Invalid RDF term: {term!r}")
    if term_type == "literal":
        datatype = term.get("datatype")
        lang = term.get("lang")
        if datatype is not None and not isinstance(datatype, str):
            raise ValueError("Literal datatype must be a string")
        if lang is not None and not isinstance(lang, str):
            raise ValueError("Literal language must be a string")
        return ("literal", value, datatype or "", (lang or "").lower())
    return (term_type, value)


def canonical_rows(
    rows: list[dict[str, Any]],
    answer_roles: list[str],
) -> set[tuple[tuple[Any, ...], ...]]:
    if not answer_roles or len(answer_roles) != len(set(answer_roles)):
        raise ValueError("answer_roles must be a nonempty unique list")
    result = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each answer row must be an object")
        missing = set(answer_roles) - set(row)
        if missing:
            raise ValueError(f"Answer row is missing roles: {sorted(missing)}")
        result.add(tuple(canonical_term(row[role]) for role in answer_roles))
    return result
