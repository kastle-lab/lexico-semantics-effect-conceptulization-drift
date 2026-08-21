#!/usr/bin/env python3
"""Compute the three paper-facing lexical/semantic alignment metrics.

This evaluator inspects only the frozen existing SPARQL artifacts. It does not
generate or execute SPARQL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import numpy as np
import pandas as pd
from rdflib import RDF, RDFS, URIRef
from rdflib.paths import AlternativePath, InvPath, MulPath, SequencePath
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue

from final_eval_common import (
    ARTIFACTS_PATH,
    REQUIREMENTS_PATH,
    ROOT,
    SCHEMA_ELEMENTS_PATH,
    sha256_file,
    sha256_text,
)


OUT_DIR = ROOT / "analysis/paper_alignment"
CSV_OUT = OUT_DIR / "artifact_paper_alignment.csv"

VERSION = "0.1.0"
EXPECTED_ARTIFACT_COUNT = 594
MODEL_ID = "Qwen/Qwen3-Embedding-4B"
LOCAL_TINY_MODEL_ID = "local-tiny-hash"
# Latest model-repository revision when this protocol was frozen.
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION = (
    "Given a competency question or semantic requirement, retrieve the "
    "ontology class or property label that best expresses its meaning."
)

LEXICAL_METRIC = "lexical_cq_selected_schema_label_jaccard"
GLOBAL_SEMANTIC_METRIC = "semantic_global_cq_selected_schema_similarity"
SOFT_SEMANTIC_METRIC = "semantic_requirement_soft_coverage"
PAPER_METRICS = [
    LEXICAL_METRIC,
    GLOBAL_SEMANTIC_METRIC,
    SOFT_SEMANTIC_METRIC,
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
DIGIT_LEFT_RE = re.compile(r"(?<=[A-Za-z])(?=[0-9])")
DIGIT_RIGHT_RE = re.compile(r"(?<=[0-9])(?=[A-Za-z])")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

# These standard predicates carry domain-relevant naming semantics but do not
# appear in the KG-local class/property inventory. rdf:type is intentionally
# absent: its class object is selected instead of the generic predicate.
STANDARD_SEMANTIC_PREDICATES = {
    str(RDFS.label): {
        "label": "label",
        "kind": "standard_label_property",
    },
    "http://www.w3.org/2004/02/skos/core#prefLabel": {
        "label": "preferred label",
        "kind": "standard_label_property",
    },
    "http://xmlns.com/foaf/0.1/name": {
        "label": "name",
        "kind": "standard_label_property",
    },
}

PROTOCOL = f"""# Paper Alignment Metric Protocol

## Boundary

This evaluator computes exactly three paper-facing alignment outcomes over the 594
frozen existing-query artifacts. It generates no SPARQL and executes no query.

## Safe schema-term recovery

SPARQL is parsed and translated to RDFLib algebra. A URI is retained only when:

1. it is a declared class or property in that KG's frozen prompt-schema
   inventory, including URIs constrained through `VALUES`, `BIND`, or filters;
2. or it is an audited naming predicate (`rdfs:label`, `skos:prefLabel`, or
   `foaf:name`) used in predicate position.

Entity constants, literals, variable names, projected variable names, SPARQL
keywords, and undeclared URI local names are excluded. For `rdf:type`, the
declared class object is retained and the generic `rdf:type` predicate is not.

The lexical form is the inventory's preferred `rdfs:label`, with split URI
local name only as a documented fallback. Preferred labels describe ontology
elements; they are not labels retrieved at query execution time.

## Missingness

A parse failure is not scored. A parsed query with no safely recoverable schema
term is `not_applicable_no_recoverable_schema_term`; it is not assigned zero.
This prevents generic variable-heavy queries from being penalized merely for
using variables, while also preventing unsupported evidence from receiving an
alignment score.

## Metrics

- `{LEXICAL_METRIC}`: set Jaccard over frozen content-token rules.
- `{GLOBAL_SEMANTIC_METRIC}`: full CQ against a concatenated selected-term
  document.
- `{SOFT_SEMANTIC_METRIC}`: required semantic-role and relation requirements
  against individual selected-term documents. `any_of` relations form one
  alternative unit; optional relations, constraints, and operators are not
  folded into this metric.

No composite, thresholded correctness label, or unconditional zero-filled
version is produced.

## Qwen configuration

Model: `{MODEL_ID}`, pinned revision `{MODEL_REVISION}`. Query-side inputs use:

`Instruct: {INSTRUCTION}\\nQuery:<text>`

Selected schema descriptions are unprompted documents. Embeddings are L2
normalized and compared by cosine similarity.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_tokens(text: str) -> set[str]:
    separated = CAMEL_RE.sub(" ", str(text).replace("_", " ").replace("-", " "))
    separated = DIGIT_LEFT_RE.sub(" ", separated)
    separated = DIGIT_RIGHT_RE.sub(" ", separated)
    tokens = {match.group(0).lower() for match in TOKEN_RE.finditer(separated)}
    return {token for token in tokens if token not in STOPWORDS}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a and b else 0.0


def uri_local_name(uri: str) -> str:
    return unquote(re.split(r"[/#]", uri.rstrip("/#"))[-1])


def load_inventory() -> dict[str, dict[str, dict[str, str]]]:
    table = pd.read_csv(SCHEMA_ELEMENTS_PATH).fillna("")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for row in table.to_dict(orient="records"):
        result.setdefault(row["kg_id"], {})[row["element_uri"]] = {
            "preferred_label": row["preferred_label"],
            "local_name": row["local_name"],
            "kind": row["element_kind"],
        }
    return result


def uris_in_path(value: Any) -> set[URIRef]:
    if isinstance(value, URIRef):
        return {value}
    if isinstance(value, (AlternativePath, SequencePath)):
        result: set[URIRef] = set()
        for item in value.args:
            result.update(uris_in_path(item))
        return result
    if isinstance(value, MulPath):
        return uris_in_path(value.path)
    if isinstance(value, InvPath):
        return uris_in_path(value.arg)
    return set()


def predicate_uris(algebra: CompValue) -> set[str]:
    result: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, CompValue):
            if value.name == "BGP":
                for _, predicate, _ in value.get("triples", []):
                    result.update(str(uri) for uri in uris_in_path(predicate))
            for nested in value.values():
                walk(nested)
        elif isinstance(value, dict):
            for key, nested in value.items():
                walk(key)
                walk(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                walk(nested)

    walk(algebra)
    return result


def all_uri_terms(value: Any) -> set[URIRef]:
    if isinstance(value, URIRef):
        return {value}
    if isinstance(value, CompValue):
        result: set[URIRef] = set()
        for nested in value.values():
            result.update(all_uri_terms(nested))
        return result
    if isinstance(value, (AlternativePath, SequencePath, MulPath, InvPath)):
        return uris_in_path(value)
    if isinstance(value, dict):
        result: set[URIRef] = set()
        for key, nested in value.items():
            result.update(all_uri_terms(key))
            result.update(all_uri_terms(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result: set[URIRef] = set()
        for nested in value:
            result.update(all_uri_terms(nested))
        return result
    return set()


def uri_terms_including_bindings(value: Any) -> set[URIRef]:
    """Traverse algebra containers, including RDFLib VALUES dictionaries."""
    if isinstance(value, URIRef):
        return {value}
    if isinstance(value, dict):
        result: set[URIRef] = set()
        for key, nested in value.items():
            result.update(uri_terms_including_bindings(key))
            result.update(uri_terms_including_bindings(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result = set()
        for nested in value:
            result.update(uri_terms_including_bindings(nested))
        return result
    return all_uri_terms(value)


def humanize_identifier(value: str) -> str:
    text = str(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return " ".join(text.split())


def requirement_units(requirement: dict[str, Any]) -> list[dict[str, Any]]:
    intent = requirement["intent"]
    units = [
        {
            "unit_id": f"role::{item['role']}",
            "category": "role",
            "alternatives": [humanize_identifier(item["concept"])],
            "source_ids": [item["role"]],
        }
        for item in intent["semantic_role_requirements"]
        if item["required"]
    ]
    any_of: dict[str, list[dict[str, Any]]] = {}
    for item in intent["relation_requirements"]:
        if item["mode"] not in {"required", "contextual_scope", "any_of"}:
            continue
        if item["mode"] == "any_of":
            group = item.get("requirement_group") or item["id"]
            any_of.setdefault(group, []).append(item)
            continue
        units.append(
            {
                "unit_id": f"relation::{item['id']}",
                "category": "relation",
                "alternatives": [
                    humanize_identifier(item["semantic_relation"])
                ],
                "source_ids": [item["id"]],
            }
        )
    for group, items in sorted(any_of.items()):
        units.append(
            {
                "unit_id": f"relation_any_of::{group}",
                "category": "relation",
                "alternatives": [
                    humanize_identifier(item["semantic_relation"])
                    for item in items
                ],
                "source_ids": [item["id"] for item in items],
            }
        )
    return units


def extract_selected_schema_terms(
    query: str,
    kg_id: str,
    inventory: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Recover schema terms without treating variables/entities as vocabulary."""
    try:
        algebra = translateQuery(parseQuery(query)).algebra
    except Exception as exc:
        return {
            "status": "not_applicable_parse_failure",
            "parse_ok": False,
            "parse_error_type": type(exc).__name__,
            "selected_terms": [],
            "excluded_uri_count": None,
            "rdf_type_present": None,
        }

    all_uris = {str(uri) for uri in uri_terms_including_bindings(algebra)}
    declared = inventory[kg_id]
    selected = []
    for uri in sorted(all_uris & set(declared)):
        item = declared[uri]
        label = item["preferred_label"] or item["local_name"] or uri_local_name(uri)
        selected.append(
            {
                "uri": uri,
                "label": label,
                "kind": item["kind"],
                "label_source": (
                    "preferred_rdfs_label"
                    if item["preferred_label"]
                    else "uri_local_name_fallback"
                ),
                "recovery_source": "declared_schema_uri_in_query_algebra",
                "document": f"{item['kind']}: {label}",
            }
        )

    predicates = predicate_uris(algebra)
    for uri, item in STANDARD_SEMANTIC_PREDICATES.items():
        if uri in predicates:
            selected.append(
                {
                    "uri": uri,
                    "label": item["label"],
                    "kind": item["kind"],
                    "label_source": "frozen_standard_predicate_label",
                    "recovery_source": "audited_standard_predicate_position",
                    "document": f"property: {item['label']}",
                }
            )

    selected = sorted(
        {item["uri"]: item for item in selected}.values(),
        key=lambda item: item["uri"],
    )
    status = (
        "eligible_safe_schema_terms"
        if selected
        else "not_applicable_no_recoverable_schema_term"
    )
    return {
        "status": status,
        "parse_ok": True,
        "parse_error_type": None,
        "selected_terms": selected,
        "excluded_uri_count": len(
            all_uris
            - set(declared)
            - set(STANDARD_SEMANTIC_PREDICATES)
            - {str(RDF.type)}
        ),
        "rdf_type_present": str(RDF.type) in predicates,
    }


def get_detailed_instruct(text: str, instruction: str = INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery:{text}"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def score_requirement_soft_coverage(
    units: list[dict[str, Any]],
    term_documents: list[str],
    query_vectors: dict[str, np.ndarray],
    document_vectors: dict[str, np.ndarray],
) -> tuple[float | None, list[dict[str, Any]]]:
    if not units:
        return None, []
    if not term_documents:
        return None, []
    details = []
    for unit in units:
        candidates = [
            {
                "requirement_alternative": alternative,
                "selected_schema_document": document,
                "similarity": cosine(
                    query_vectors[alternative],
                    document_vectors[document],
                ),
            }
            for alternative in unit["alternatives"]
            for document in term_documents
        ]
        best = max(candidates, key=lambda item: item["similarity"])
        details.append({**unit, "score": best["similarity"], "best_match": best})
    return float(np.mean([item["score"] for item in details])), details


def build_base_records() -> list[dict[str, Any]]:
    artifacts = pd.read_csv(ARTIFACTS_PATH, dtype=str).fillna("")
    requirements = {
        item["cq_id"]: item for item in read_jsonl(REQUIREMENTS_PATH)
    }
    inventory = load_inventory()
    records = []
    for row in artifacts.to_dict(orient="records"):
        evaluated_query = row.get("evaluation_query") or row["sparql_query"]
        extracted = extract_selected_schema_terms(
            evaluated_query, row["kg_id"], inventory
        )
        labels = [item["label"] for item in extracted["selected_terms"]]
        cq_tokens = normalize_tokens(row["CQ"])
        label_tokens = normalize_tokens(" ".join(labels))
        eligible = extracted["status"] == "eligible_safe_schema_terms"
        records.append(
            {
                "alignment_version": VERSION,
                "artifact_id": row["artifact_id"],
                "query_sha256": sha256_text(evaluated_query),
                "cq_id": row["cq_id"],
                "cq_index": int(row["cq_index"]),
                "difficulty": row["difficulty"],
                "kg_id": row["kg_id"],
                "model_id": row["model_id"],
                "alignment_scoring_status": extracted["status"],
                "safe_schema_term_eligible": eligible,
                "strict_parse_ok": extracted["parse_ok"],
                "parse_error_type": extracted["parse_error_type"],
                "rdf_type_present": extracted["rdf_type_present"],
                "rdf_type_handling": "class_object_only",
                "excluded_non_schema_uri_count": extracted["excluded_uri_count"],
                "cq_text": row["CQ"],
                "selected_schema_terms": extracted["selected_terms"],
                "selected_schema_term_count": len(extracted["selected_terms"]),
                "requirement_units": requirement_units(
                    requirements[row["cq_id"]]
                ),
                LEXICAL_METRIC: (
                    jaccard(cq_tokens, label_tokens) if eligible else None
                ),
                GLOBAL_SEMANTIC_METRIC: None,
                SOFT_SEMANTIC_METRIC: None,
                "semantic_requirement_scores": [],
            }
        )
    if len(records) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError(f"Expected {EXPECTED_ARTIFACT_COUNT} artifacts")
    if any(item["model_id"].lower().startswith("phi4") for item in records):
        raise ValueError("Excluded Phi-4 artifact found")
    return records


def encode_texts(
    model: Any,
    texts: list[str],
    *,
    query_side: bool,
    batch_size: int,
    instruction: str,
) -> dict[str, np.ndarray]:
    inputs = (
        [get_detailed_instruct(text, instruction) for text in texts]
        if query_side
        else texts
    )
    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {
        text: vector.astype(np.float32, copy=False)
        for text, vector in zip(texts, vectors)
    }


def tiny_hash_vector(text: str, *, dimensions: int = 256) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in normalize_tokens(text):
        digest = sha256_text(token)
        bucket = int(digest[:8], 16) % dimensions
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def add_local_tiny_scores(
    records: list[dict[str, Any]],
    *,
    model_id: str,
    revision: str,
    instruction: str,
) -> None:
    eligible = [item for item in records if item["safe_schema_term_eligible"]]
    query_texts = {item["cq_text"] for item in eligible}
    document_texts: set[str] = set()
    for item in eligible:
        query_texts.update(
            alternative
            for unit in item["requirement_units"]
            for alternative in unit["alternatives"]
        )
        term_documents = [
            term["document"] for term in item["selected_schema_terms"]
        ]
        document_texts.update(term_documents)
        document_texts.add(" ; ".join(term_documents))
    queries = {text: tiny_hash_vector(text) for text in query_texts}
    documents = {text: tiny_hash_vector(text) for text in document_texts}
    for item in eligible:
        term_documents = [
            term["document"] for term in item["selected_schema_terms"]
        ]
        combined = " ; ".join(term_documents)
        item[GLOBAL_SEMANTIC_METRIC] = cosine(
            queries[item["cq_text"]], documents[combined]
        )
        coverage, details = score_requirement_soft_coverage(
            item["requirement_units"],
            term_documents,
            queries,
            documents,
        )
        item[SOFT_SEMANTIC_METRIC] = coverage
        item["semantic_requirement_scores"] = details
        item["embedding_model_id"] = model_id
        item["embedding_model_revision"] = revision
        item["embedding_query_instruction"] = instruction


def add_semantic_scores(
    records: list[dict[str, Any]],
    *,
    model_id: str,
    revision: str,
    instruction: str,
    device: str,
    batch_size: int,
    cache_dir: Path | None,
) -> None:
    if model_id == LOCAL_TINY_MODEL_ID:
        add_local_tiny_scores(
            records,
            model_id=model_id,
            revision=revision,
            instruction=instruction,
        )
        return
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from sentence_transformers import SentenceTransformer

    eligible = [item for item in records if item["safe_schema_term_eligible"]]
    query_texts = {
        item["cq_text"] for item in eligible
    }
    document_texts: set[str] = set()
    for item in eligible:
        query_texts.update(
            alternative
            for unit in item["requirement_units"]
            for alternative in unit["alternatives"]
        )
        term_documents = [
            term["document"] for term in item["selected_schema_terms"]
        ]
        document_texts.update(term_documents)
        document_texts.add(" ; ".join(term_documents))

    model = SentenceTransformer(
        model_id,
        revision=revision,
        cache_folder=str(cache_dir) if cache_dir else None,
        device=None if device == "auto" else device,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": "bfloat16"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    queries = encode_texts(
        model,
        sorted(query_texts),
        query_side=True,
        batch_size=batch_size,
        instruction=instruction,
    )
    documents = encode_texts(
        model,
        sorted(document_texts),
        query_side=False,
        batch_size=batch_size,
        instruction=instruction,
    )
    for item in eligible:
        term_documents = [
            term["document"] for term in item["selected_schema_terms"]
        ]
        combined = " ; ".join(term_documents)
        item[GLOBAL_SEMANTIC_METRIC] = cosine(
            queries[item["cq_text"]], documents[combined]
        )
        coverage, details = score_requirement_soft_coverage(
            item["requirement_units"],
            term_documents,
            queries,
            documents,
        )
        item[SOFT_SEMANTIC_METRIC] = coverage
        item["semantic_requirement_scores"] = details
        item["embedding_model_id"] = model_id
        item["embedding_model_revision"] = revision
        item["embedding_query_instruction"] = instruction


def flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = dict(record)
        for field in (
            "selected_schema_terms",
            "requirement_units",
            "semantic_requirement_scores",
        ):
            row[field] = json.dumps(
                row[field], ensure_ascii=True, separators=(",", ":")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def preserve_existing_semantic_scores(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    """Keep prior Qwen scores only when their exact scoring inputs match."""
    if not CSV_OUT.exists():
        return 0
    existing = pd.read_csv(CSV_OUT, low_memory=False)
    required = {
        "artifact_id",
        "selected_schema_terms",
        "requirement_units",
        GLOBAL_SEMANTIC_METRIC,
        SOFT_SEMANTIC_METRIC,
        "semantic_requirement_scores",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_query_instruction",
    }
    if not required.issubset(existing.columns):
        return 0
    existing = existing.set_index("artifact_id")
    preserved = 0
    for record in records:
        artifact_id = record["artifact_id"]
        if artifact_id not in existing.index:
            continue
        prior = existing.loc[artifact_id]
        if isinstance(prior, pd.DataFrame):
            raise ValueError(f"Duplicate prior alignment row: {artifact_id}")
        selected = json.dumps(
            record["selected_schema_terms"],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        requirements = json.dumps(
            record["requirement_units"],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        metadata_matches = (
            str(prior["embedding_model_id"]) == args.model_id
            and str(prior["embedding_model_revision"]) == args.model_revision
            and str(prior["embedding_query_instruction"]) == args.instruction
        )
        inputs_match = (
            str(prior["selected_schema_terms"]) == selected
            and str(prior["requirement_units"]) == requirements
        )
        scores_exist = pd.notna(prior[GLOBAL_SEMANTIC_METRIC]) and pd.notna(
            prior[SOFT_SEMANTIC_METRIC]
        )
        if not (metadata_matches and inputs_match and scores_exist):
            continue
        record[GLOBAL_SEMANTIC_METRIC] = float(
            prior[GLOBAL_SEMANTIC_METRIC]
        )
        record[SOFT_SEMANTIC_METRIC] = float(prior[SOFT_SEMANTIC_METRIC])
        record["semantic_requirement_scores"] = json.loads(
            prior["semantic_requirement_scores"]
        )
        record["embedding_model_id"] = args.model_id
        record["embedding_model_revision"] = args.model_revision
        record["embedding_query_instruction"] = args.instruction
        preserved += 1
    return preserved


def validate_semantic_scores(records: list[dict[str, Any]]) -> None:
    eligible = [item for item in records if item["safe_schema_term_eligible"]]
    incomplete = [
        item["artifact_id"]
        for item in eligible
        if item[GLOBAL_SEMANTIC_METRIC] is None
        or item[SOFT_SEMANTIC_METRIC] is None
    ]
    if incomplete:
        raise ValueError(
            "Full semantic scoring left eligible artifacts incomplete: "
            f"{len(incomplete)}; examples={incomplete[:5]}"
        )


def write_outputs(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    preserved_semantic_count: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = flatten(records)
    table.to_csv(CSV_OUT, index=False)
    manifest = {
        "version": VERSION,
        "artifact_count": len(records),
        "paper_metrics": PAPER_METRICS,
        "generated_sparql_count": 0,
        "executed_query_count": 0,
        "semantic_scores_computed": not args.lexical_only,
        "semantic_scores_preserved": preserved_semantic_count,
        "semantic_global_non_null_count": sum(
            item[GLOBAL_SEMANTIC_METRIC] is not None for item in records
        ),
        "semantic_soft_non_null_count": sum(
            item[SOFT_SEMANTIC_METRIC] is not None for item in records
        ),
        "embedding_model_id": args.model_id,
        "embedding_model_revision": args.model_revision,
        "embedding_query_instruction": args.instruction,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ARTIFACTS_PATH,
                REQUIREMENTS_PATH,
                SCHEMA_ELEMENTS_PATH,
            )
        },
    }
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-cache-dir", type=Path)
    args = parser.parse_args()

    records = build_base_records()
    preserved_semantic_count = 0
    if not args.lexical_only:
        add_semantic_scores(
            records,
            model_id=args.model_id,
            revision=args.model_revision,
            instruction=args.instruction,
            device=args.device,
            batch_size=args.batch_size,
            cache_dir=args.model_cache_dir,
        )
        validate_semantic_scores(records)
    else:
        preserved_semantic_count = preserve_existing_semantic_scores(
            records, args
        )
    write_outputs(
        records,
        args,
        preserved_semantic_count=preserved_semantic_count,
    )


if __name__ == "__main__":
    main()
