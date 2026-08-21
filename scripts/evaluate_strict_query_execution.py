#!/usr/bin/env python3
"""Recompute execution diagnostics for exact strict frozen queries."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from final_eval_common import ARTIFACTS_PATH, EXECUTION_METRICS_PATH, sha256_text
from hypothesis_specific_eval import (
    determinism_score_from_flag,
    satisfiability_binding_score_from_flags,
)
from sparql_eval_module import (
    _LocalTinyHash,
    _SBERT,
    _cosine,
    _is_uri,
    _norm_text,
    _project_vars,
    _result_checksum,
    _run,
    _tail,
)
from rdflib.plugins.sparql.parser import parseQuery


VERSION = "1.0.0"
EXECUTION_FIELDS = [
    "satisfiable",
    "deterministic",
    "rows",
    "variables",
    "always_unbound_vars",
    "determinism_score",
    "satisfiability_binding_score",
    "tuple_cohesion",
]
PROVENANCE_FIELDS = [
    "execution_query_sha256",
    "execution_kg_snapshot_id",
    "execution_repetitions",
    "execution_evaluator_version",
]
QUERY_FORM_RE = re.compile(
    r"(?im)^[ \t]*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b"
)
EXECUTION_COLUMNS = [
    "artifact_id",
    "artifact_status",
    "syntax_ok",
    "query_form",
    "satisfiable",
    "deterministic",
    "rows",
    "variables",
    "always_unbound_vars",
    "determinism_score",
    "satisfiability_binding_score",
    "tuple_cohesion",
    "execution_query_sha256",
    "execution_kg_snapshot_id",
    "execution_repetitions",
    "execution_evaluator_version",
]


def parse_mapping(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use KG_ID=value")
        key, mapped = value.split("=", 1)
        if not key.strip() or not mapped.strip():
            raise ValueError(f"Invalid {label}: {value}")
        result[key.strip()] = mapped.strip()
    return result


def stored_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def syntax_ok(query: str) -> bool:
    if not str(query).strip():
        return False
    try:
        parseQuery(str(query))
        return True
    except Exception:
        return False


def query_form(query: str) -> str:
    match = QUERY_FORM_RE.search(str(query))
    return match.group(1).title() + "Query" if match else ""


def base_execution_rows(artifacts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for artifact in artifacts.to_dict(orient="records"):
        evaluation_query = str(
            artifact.get("evaluation_query") or artifact["sparql_query"]
        )
        valid = syntax_ok(evaluation_query)
        if not str(artifact["sparql_query"]).strip():
            status = (
                "NO_QUERY"
                if artifact.get("query_cleaning_status") == "empty_model_output"
                else "NON_SPARQL_OUTPUT"
            )
        elif not valid:
            status = "PARSE_FAILURE"
        else:
            status = "PENDING_STRICT_EXECUTION"
        rows.append(
            {
                "artifact_id": artifact["artifact_id"],
                "artifact_status": status,
                "syntax_ok": valid,
                "query_form": query_form(evaluation_query),
                "satisfiable": False if not valid else pd.NA,
                "deterministic": False if not valid else pd.NA,
                "rows": 0 if not valid else pd.NA,
                "variables": "[]",
                "always_unbound_vars": "[]",
                "determinism_score": 0.0 if not valid else pd.NA,
                "satisfiability_binding_score": 0.0 if not valid else pd.NA,
                "tuple_cohesion": pd.NA,
                "execution_query_sha256": "",
                "execution_kg_snapshot_id": "",
                "execution_repetitions": "",
                "execution_evaluator_version": "",
            }
        )
    return pd.DataFrame(rows, columns=EXECUTION_COLUMNS)


def clear_execution(row: dict[str, Any]) -> None:
    for field in EXECUTION_FIELDS:
        row[field] = pd.NA


def evaluate_artifact(
    artifact: dict[str, Any],
    endpoint: str,
    embedder: Any,
    *,
    snapshot_id: str,
    repetitions: int,
    max_result_rows: int | None,
) -> dict[str, Any]:
    """Evaluate one exact query and return fields to merge into its row."""
    query = str(artifact.get("evaluation_query") or artifact["sparql_query"])
    if not query.strip() or not stored_bool(artifact["syntax_ok"]):
        raise ValueError("Only nonempty strictly parsed queries are executable")

    executed_query = locally_limited_query(query, max_result_rows)
    runs = [_run(endpoint, executed_query)[0] for _ in range(repetitions)]
    first = runs[0] if runs else []
    stable = len({_result_checksum(rows) for rows in runs}) == 1
    variable_names = _project_vars(query)
    if first:
        variable_names = variable_names or sorted(
            {variable for row in first for variable in row}
        )
    always_unbound = [
        variable
        for variable in variable_names
        if not any(variable in row for row in first)
    ]
    variables = [{"var": variable} for variable in variable_names]
    satisfiable = bool(first)
    tuple_cohesion = tuple_cohesion_from_rows(first, variable_names, embedder)
    updates = {
        "artifact_status": "EXECUTABLE",
        "satisfiable": satisfiable,
        "deterministic": stable,
        "rows": len(first),
        "variables": json.dumps(variables, ensure_ascii=True),
        "always_unbound_vars": json.dumps(always_unbound, ensure_ascii=True),
        "determinism_score": determinism_score_from_flag(stable),
        "satisfiability_binding_score": (
            satisfiability_binding_score_from_flags(
                satisfiable, always_unbound, len(variables)
            )
        ),
        "tuple_cohesion": tuple_cohesion,
        "execution_query_sha256": sha256_text(executed_query),
        "execution_kg_snapshot_id": snapshot_id,
        "execution_repetitions": repetitions,
        "execution_evaluator_version": (
            f"{VERSION}+local_result_limit_{max_result_rows}"
            if max_result_rows
            else VERSION
        ),
    }
    return updates


def make_embedder(model_id: str | None) -> Any:
    return _LocalTinyHash() if model_id == "local-tiny-hash" else _SBERT(model_id)


def locally_limited_query(query: str, max_result_rows: int | None) -> str:
    if not max_result_rows or max_result_rows < 1:
        return query
    if not query.lstrip().upper().startswith(("PREFIX", "BASE", "SELECT")):
        return query
    if "LIMIT" in query.upper():
        return query
    return query.rstrip().rstrip(";") + f"\nLIMIT {max_result_rows}"


def tuple_cohesion_from_rows(
    rows: list[dict[str, Any]], variables: list[str], embedder: Any
) -> float | None:
    if not rows or not variables:
        return None
    tuples = []
    for row in rows:
        parts = []
        for variable in variables:
            if variable not in row:
                parts.append("")
                continue
            binding = row[variable]
            raw = binding["value"]
            if _is_uri(binding):
                parts.append(_norm_text(_tail(raw)))
            else:
                if "xml:lang" in binding:
                    raw += f"@{binding['xml:lang']}"
                if "datatype" in binding:
                    raw += f"^^<{binding['datatype']}>"
                parts.append(_norm_text(raw))
        tuples.append(" | ".join(parts))
    unique = list(dict.fromkeys(tuples))
    max_values = int(os.environ.get("SPARQL_EVAL_MAX_COHESION_VALUES", "0") or 0)
    if max_values > 0:
        unique = unique[:max_values]
    if len(unique) <= 1:
        return 1.0
    embeddings = embedder.embed(unique)
    if embeddings.shape[0] <= 1:
        return 1.0
    similarities = _cosine(embeddings, embeddings)
    import numpy as np

    np.fill_diagonal(similarities, -1.0)
    value = float(similarities.max(axis=1).mean())
    return round(value, 2) if value else value


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".execution.tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="KG_ID=https://... SPARQL endpoint; repeat per KG.",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        help="KG_ID=immutable-snapshot-id; repeat per KG.",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--artifact-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--endpoint-timeout-seconds", type=int)
    parser.add_argument("--max-label-uris", type=int)
    parser.add_argument("--max-cohesion-values", type=int)
    parser.add_argument("--max-result-rows", type=int)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Recompute already executable strict queries as well as pending ones.",
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.endpoint_timeout_seconds is not None:
        os.environ["SPARQL_EVAL_TIMEOUT_SECONDS"] = str(
            args.endpoint_timeout_seconds
        )
    if args.max_label_uris is not None:
        os.environ["SPARQL_EVAL_MAX_LABEL_URIS"] = str(args.max_label_uris)
    if args.max_cohesion_values is not None:
        os.environ["SPARQL_EVAL_MAX_COHESION_VALUES"] = str(
            args.max_cohesion_values
        )

    endpoints = parse_mapping(args.endpoint, "endpoint")
    snapshots = parse_mapping(args.snapshot, "snapshot")
    if not endpoints or set(endpoints) != set(snapshots):
        parser.error("endpoint and snapshot KG IDs must match and be nonempty")

    artifacts = pd.read_csv(
        ARTIFACTS_PATH, keep_default_na=False, low_memory=False
    )
    if len(artifacts) != 594:
        raise ValueError(f"Expected 594 frozen artifacts, found {len(artifacts)}")
    execution = base_execution_rows(artifacts)
    artifacts_for_execution = artifacts.merge(
        execution[["artifact_id", "artifact_status", "syntax_ok", "query_form"]],
        on="artifact_id",
        how="left",
        validate="one_to_one",
    )
    runnable = execution["syntax_ok"].map(stored_bool)
    runnable &= artifacts["sparql_query"].astype(str).str.strip().ne("")
    eligible = runnable.copy()
    eligible &= artifacts["kg_id"].isin(endpoints)
    if args.artifact_id:
        eligible &= artifacts["artifact_id"].isin(args.artifact_id)
    selected = list(artifacts.index[eligible])
    if args.limit is not None:
        selected = selected[: args.limit]

    embedder = make_embedder(args.model_id)
    failures: list[dict[str, str]] = []
    completed = 0
    for position, index in enumerate(selected, start=1):
        artifact = artifacts_for_execution.loc[index].to_dict()
        kg_id = str(artifact["kg_id"])
        try:
            updates = evaluate_artifact(
                artifact,
                endpoints[kg_id],
                embedder,
                snapshot_id=snapshots[kg_id],
                repetitions=args.repetitions,
                max_result_rows=args.max_result_rows,
            )
            for field, value in updates.items():
                execution.at[index, field] = value
            completed += 1
        except Exception as exc:
            execution.at[index, "artifact_status"] = "EXECUTION_FAILURE"
            evaluation_query = str(
                artifact.get("evaluation_query") or artifact["sparql_query"]
            )
            execution.at[index, "satisfiable"] = False
            execution.at[index, "deterministic"] = False
            execution.at[index, "rows"] = 0
            execution.at[index, "variables"] = "[]"
            execution.at[index, "always_unbound_vars"] = "[]"
            execution.at[index, "determinism_score"] = 0.0
            execution.at[index, "satisfiability_binding_score"] = 0.0
            execution.at[index, "execution_query_sha256"] = sha256_text(
                evaluation_query
            )
            execution.at[index, "execution_kg_snapshot_id"] = snapshots[kg_id]
            execution.at[index, "execution_repetitions"] = args.repetitions
            execution.at[index, "execution_evaluator_version"] = VERSION
            failures.append(
                {
                    "artifact_id": str(artifact["artifact_id"]),
                    "error": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )
        if position % args.checkpoint_every == 0:
            EXECUTION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(execution, EXECUTION_METRICS_PATH)

    EXECUTION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(execution, EXECUTION_METRICS_PATH)
    print(
        json.dumps(
            {
                "full_population_artifact_count": len(artifacts),
                "selected_artifact_count": len(selected),
                "pre_execution_failure_count": int((~runnable).sum()),
                "completed_execution_count": completed,
                "execution_failure_count": len(failures),
                "failure_examples": failures[:10],
                "repetitions": args.repetitions,
                "generated_query_count": 0,
                "schema_prefix_normalization_retained": True,
                "executed_exact_frozen_queries": True,
                "output": str(EXECUTION_METRICS_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
