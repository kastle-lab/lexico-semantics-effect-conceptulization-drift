#!/usr/bin/env python3
"""Clean wrappers and build prefix-normalized copies without changing query bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib.plugins.sparql.parser import parseQuery

from final_eval_common import ARTIFACTS_PATH, ROOT, sha256_file


QUERY_FORM_RE = re.compile(
    r"(?im)^[ \t]*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b"
)
QUERY_START_RE = re.compile(
    r"(?i)^(?:PREFIX\s+[A-Za-z][\w-]*\s*:|BASE\s+<|"
    r"SELECT\b|ASK\b|CONSTRUCT\b|DESCRIBE\b)"
)
PREFIX_LINE_RE = re.compile(
    r"(?i)^[ \t]*(?:PREFIX\s+[A-Za-z][\w-]*\s*:\s*<[^\r\n>]*>"
    r"|BASE\s+<[^\r\n>]*>)[ \t]*$"
)
FENCE_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.I | re.S)
COMPLETE_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
OPEN_THINK_RE = re.compile(r"<think>", re.I)
CLOSE_THINK_RE = re.compile(r"</think>", re.I)
MANIFEST_COLUMNS = [
    "artifact_id",
    "cq_id",
    "cq_index",
    "difficulty",
    "kg_id",
    "kg_name",
    "model_id",
    "source_file",
    "source_sheet",
    "source_excel_row",
    "source_file_sha256",
    "CQ",
    "sparql_query",
    "evaluation_query",
    "legacy_query_sha256",
    "model_output_sha256",
    "strict_query_sha256",
    "evaluation_query_sha256",
    "query_cleaning_status",
    "query_changed_from_legacy",
    "query_normalization_status",
    "added_schema_prefixes",
    "prefix_source_sha256",
]
PREFIX_TTL_BY_KG = {
    "marveldb_big": ROOT / "datasets/ttl_for_prefix_repair/big_schema_for_prefix.ttl",
    "mcuwiki_small": ROOT / "datasets/ttl_for_prefix_repair/small_schema_for_prefix.ttl",
}
KG_PREFIX_ALIASES = {
    "marveldb_big": {
        "marvel": "http://dbkwik.webdatacommons.org/marvel.wikia.com/resource/",
    },
    "mcuwiki_small": {
        "marvel": (
            "http://dbkwik.webdatacommons.org/"
            "marvelcinematicuniverse.wikia.com/resource/"
        ),
        "marvel-class": (
            "http://dbkwik.webdatacommons.org/"
            "marvelcinematicuniverse.wikia.com/class/"
        ),
    },
}
TTL_PREFIX_RE = re.compile(
    r"(?im)^\s*@prefix\s+([A-Za-z][A-Za-z0-9_-]*):\s*<([^>]+)>\s*\."
)
DECLARED_PREFIX_RE = re.compile(
    r"(?i)(?:@prefix|prefix)\s+([A-Za-z][A-Za-z0-9_-]*):\s*<([^>]+)>"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stored_bool(value: Any, default: bool) -> bool:
    """Read booleans already serialized through CSV without truthy-string bugs."""
    if pd.isna(value) or value == "":
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return bool(value)


@lru_cache(maxsize=8192)
def syntax_valid(query: str) -> bool:
    if not query.strip():
        return False
    try:
        parseQuery(query)
        return True
    except Exception:
        return False


@lru_cache(maxsize=4)
def ttl_prefixes(kg_id: str) -> dict[str, str]:
    path = PREFIX_TTL_BY_KG[kg_id]
    text = path.read_text(encoding="utf-8")
    prefixes = dict(TTL_PREFIX_RE.findall(text))
    for name, namespace in KG_PREFIX_ALIASES.get(kg_id, {}).items():
        if namespace in text:
            prefixes.setdefault(name, namespace)
    return prefixes


def add_missing_prefixes(
    query: str, prefixes: dict[str, str]
) -> tuple[str, list[str]]:
    """Add every missing schema-declared PREFIX line, preserving the query body."""
    if not query.strip():
        return "", []
    existing = {match.group(1) for match in DECLARED_PREFIX_RE.finditer(query)}
    added = [name for name in prefixes if name not in existing]
    if not added:
        return query, []
    declarations = [f"PREFIX {name}: <{prefixes[name]}>" for name in added]
    return "\n".join(declarations) + "\n" + query.lstrip(), added


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        spans.append((offset, offset + len(line), line.rstrip("\r\n")))
        offset += len(line)
    if not spans or offset < len(text):
        spans.append((offset, len(text), text[offset:]))
    return spans


def _candidate_start(
    spans: list[tuple[int, int, str]], form_start: int
) -> int:
    line_index = next(
        index
        for index, (start, end, _) in enumerate(spans)
        if start <= form_start < end
    )
    start_index = line_index
    index = line_index - 1
    saw_prefix = False
    while index >= 0:
        line = spans[index][2]
        if not line.strip():
            if saw_prefix:
                start_index = index
            index -= 1
            continue
        if PREFIX_LINE_RE.fullmatch(line):
            saw_prefix = True
            start_index = index
            index -= 1
            continue
        break
    return spans[start_index][0]


def parsed_plain_candidates(text: str) -> list[str]:
    """Return distinct contiguous substrings that parse as one query."""
    spans = _line_spans(text)
    by_start: dict[int, set[str]] = {}
    for match in QUERY_FORM_RE.finditer(text):
        start = _candidate_start(spans, match.start())
        candidates = by_start.setdefault(start, set())
        for _, end, _ in reversed(spans):
            if end <= match.end():
                continue
            candidate = text[start:end].strip()
            if syntax_valid(candidate):
                candidates.add(candidate)
                break
    if not by_start:
        return []
    longest = [
        max(candidates, key=len)
        for candidates in by_start.values()
        if candidates
    ]
    return list(dict.fromkeys(longest))


def explicit_invalid_query(text: str) -> str:
    """Retain a bounded invalid query only when the output starts as SPARQL."""
    stripped = text.strip()
    if not QUERY_START_RE.match(stripped):
        return ""
    forms = list(QUERY_FORM_RE.finditer(stripped))
    if len(forms) != 1:
        return ""
    last_close = stripped.rfind("}")
    if last_close >= 0 and stripped[last_close + 1 :].strip():
        return ""
    return stripped


def query_from_region(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if syntax_valid(stripped):
        return stripped, "exact_query"
    parsed = parsed_plain_candidates(text)
    if len(parsed) == 1:
        candidate = parsed[0]
        status = (
            "exact_query"
            if candidate == text.strip()
            else "prose_wrapper_removed"
        )
        return candidate, status
    if len(parsed) > 1:
        return "", "ambiguous_multiple_queries"
    invalid = explicit_invalid_query(text)
    if invalid:
        return invalid, "explicit_invalid_query"
    return "", "non_sparql_output"


def extract_strict_query(raw_output: str) -> tuple[str, str]:
    """Extract one unchanged query substring under conservative rules."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return "", "empty_model_output"

    working = raw_output
    think_filtered = bool(COMPLETE_THINK_RE.search(working))
    working = COMPLETE_THINK_RE.sub("", working)
    opens = len(OPEN_THINK_RE.findall(working))
    closes = len(CLOSE_THINK_RE.findall(working))
    if opens > closes:
        return "", "unterminated_think_block"
    if closes > opens:
        working = CLOSE_THINK_RE.split(working)[-1]
        think_filtered = True

    fenced_candidates = []
    for block in FENCE_RE.findall(working):
        query, status = query_from_region(block)
        if query:
            fenced_candidates.append(query)
        elif status == "ambiguous_multiple_queries":
            return "", status
    fenced_candidates = list(dict.fromkeys(fenced_candidates))
    if len(fenced_candidates) == 1:
        status = "think_filtered_fenced_query" if think_filtered else "fenced_query"
        return fenced_candidates[0], status
    if len(fenced_candidates) > 1:
        return "", "ambiguous_multiple_queries"

    without_fences = FENCE_RE.sub("", working)
    query, status = query_from_region(without_fences)
    if think_filtered and query:
        status = "think_filtered_" + status
    elif think_filtered and status == "non_sparql_output":
        status = "think_filtered_no_query"
    return query, status


def source_model_output(
    artifact: dict[str, Any], cache: dict[tuple[str, str], tuple[pd.DataFrame, str]]
) -> str:
    source = ROOT / str(artifact["source_file"])
    key = (str(source), str(artifact["source_sheet"]))
    if key not in cache:
        expected_hash = str(artifact["source_file_sha256"])
        if sha256_file(source) != expected_hash:
            raise ValueError(f"Source workbook hash changed: {source}")
        frame = pd.read_excel(source, sheet_name=artifact["source_sheet"], keep_default_na=False)
        result_columns = [
            column for column in frame.columns if str(column).endswith("_Result")
        ]
        if len(result_columns) != 1:
            raise ValueError(
                f"Expected one model result column in {source}: {result_columns}"
            )
        cache[key] = (frame, result_columns[0])
    frame, result_column = cache[key]
    row_index = int(artifact["source_excel_row"]) - 2
    return str(frame.iloc[row_index][result_column])


def clean_artifacts(artifacts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    for artifact in artifacts.to_dict(orient="records"):
        raw = source_model_output(artifact, cache)
        current_query = str(artifact.get("sparql_query", ""))
        query, cleaning_status = extract_strict_query(raw)
        evaluation_query, added_prefixes = add_missing_prefixes(
            query, ttl_prefixes(str(artifact["kg_id"]))
        )
        valid = syntax_valid(evaluation_query)
        changed_now = query.strip() != current_query.strip()
        prior_legacy_hash = artifact.get("legacy_query_sha256", "")
        previously_cleaned = isinstance(prior_legacy_hash, str) and bool(
            prior_legacy_hash.strip()
        )
        changed_from_legacy = stored_bool(
            artifact.get("query_changed_from_legacy"), changed_now
        ) if previously_cleaned else changed_now

        artifact["legacy_query_sha256"] = (
            prior_legacy_hash if previously_cleaned else sha256_text(current_query)
        )
        artifact["model_output_sha256"] = sha256_text(raw)
        artifact["strict_query_sha256"] = sha256_text(query)
        artifact["evaluation_query"] = evaluation_query
        artifact["evaluation_query_sha256"] = sha256_text(evaluation_query)
        artifact["query_normalization_status"] = (
            "schema_prefixes_added"
            if added_prefixes
            else ("unchanged" if query else "no_query")
        )
        artifact["added_schema_prefixes"] = json.dumps(added_prefixes)
        artifact["prefix_source_sha256"] = (
            sha256_file(PREFIX_TTL_BY_KG[str(artifact["kg_id"])])
            if query
            else ""
        )
        artifact["query_cleaning_status"] = cleaning_status
        artifact["query_changed_from_legacy"] = changed_from_legacy
        artifact["sparql_query"] = query
        rows.append(artifact)

    cleaned = pd.DataFrame(rows).loc[:, MANIFEST_COLUMNS]
    syntax_ok = cleaned["evaluation_query"].fillna("").map(syntax_valid)
    artifact_status = []
    for row, valid in zip(cleaned.to_dict(orient="records"), syntax_ok, strict=True):
        if not str(row["sparql_query"]).strip():
            status = (
                "NO_QUERY"
                if row["query_cleaning_status"] == "empty_model_output"
                else "NON_SPARQL_OUTPUT"
            )
        elif not valid:
            status = "PARSE_FAILURE"
        else:
            status = "PENDING_STRICT_EXECUTION"
        artifact_status.append(status)
    summary = {
        "artifact_count": len(cleaned),
        "changed_from_legacy_count": int(
            cleaned["query_changed_from_legacy"].sum()
        ),
        "syntax_valid_count": int(syntax_ok.sum()),
        "query_present_count": int(cleaned["sparql_query"].str.strip().ne("").sum()),
        "cleaning_status_counts": cleaned["query_cleaning_status"].value_counts().to_dict(),
        "artifact_status_counts": pd.Series(artifact_status).value_counts().to_dict(),
        "generated_query_count": 0,
        "schema_prefix_normalized_count": int(
            cleaned["query_normalization_status"]
            .eq("schema_prefixes_added")
            .sum()
        ),
    }
    return cleaned, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace the frozen artifact table after auditing.",
    )
    parser.add_argument("--model", help="Audit one generator model only.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the audited rows to an explicit CSV without replacing input.",
    )
    args = parser.parse_args()
    artifacts = pd.read_csv(ARTIFACTS_PATH, keep_default_na=False, low_memory=False)
    if args.model:
        artifacts = artifacts.loc[artifacts["model_id"].eq(args.model)].copy()
        if artifacts.empty:
            raise ValueError(f"Unknown model: {args.model}")
    if args.write and args.model:
        raise ValueError("--write cannot be combined with --model")
    cleaned, summary = clean_artifacts(artifacts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(args.output, index=False)
        summary["output"] = str(args.output)
    if args.write:
        temporary = ARTIFACTS_PATH.with_suffix(".strict-cleaning.tmp.csv")
        cleaned.to_csv(temporary, index=False)
        temporary.replace(ARTIFACTS_PATH)
        summary["written_to"] = str(ARTIFACTS_PATH)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
