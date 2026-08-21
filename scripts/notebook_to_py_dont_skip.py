from compute_paper_alignment_metrics import (
    REQUIREMENTS_PATH,
    LEXICAL_METRIC,
    GLOBAL_SEMANTIC_METRIC,
    SOFT_SEMANTIC_METRIC,
    INSTRUCTION,
    MODEL_REVISION,
    read_jsonl,
    load_inventory,
    extract_selected_schema_terms,
    normalize_tokens,
    jaccard,
    requirement_units,
    add_semantic_scores,
)
from evaluate_strict_query_execution import syntax_ok as strict_syntax_ok, query_form
from clean_frozen_query_artifacts import (
    extract_strict_query,
    ttl_prefixes as schema_ttl_prefixes,
    add_missing_prefixes as add_schema_missing_prefixes,
    sha256_text,
)
from hypothesis_specific_eval import add_hyp_scores
from sparql_eval_module import SingleGraphCQEvaluator
import os
import re
import sys
import json
import concurrent.futures
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR if (
    NOTEBOOK_DIR / "scripts").exists() else NOTEBOOK_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================# ============================================================
# 1. Initialize configuration
# ============================================================
END_POINT = None
evaluator = None
graph_id = None
input_file = None

# Use the same labels as the original notebook. The KG ids are the ids used by
# the newer metric helpers and dependency files.
KG_ID_BY_GRAPH = {
    "Small": "mcuwiki_small",
    "Big": "marveldb_big",
}

END_POINT_DICT = {
    "Small": "http://arsenal.cs.wright.edu:3030/temp_chris_thesis1/query",
    "Big":  "http://arsenal.cs.wright.edu:3030/temp_chris_thesis2/query",
}

SNAPSHOT_ID_DICT = {
    "Small": "arsenal_temp_chris_thesis1_2026-07-31_local_test",
    "Big": "arsenal_temp_chris_thesis2_2026-07-31_local_test",
}

# Local smoke tests should use local-tiny-hash. Final HPC/paper runs should use:
SEMANTIC_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
SEMANTIC_MODEL_REVISION = None
# SEMANTIC_MODEL_ID = "local-tiny-hash"
# SEMANTIC_MODEL_REVISION = "local-test"
SEMANTIC_DEVICE = "cuda"
SEMANTIC_BATCH_SIZE = 1
SEMANTIC_CACHE_DIR = None
TUPLE_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

EXECUTION_TIMEOUT_SECONDS = 1800
RUN_LEGACY_EVALUATOR = True
RUN_ALIGNMENT_METRICS = True
EXPORT_RUBRIC_WORKBOOK = True
RUN_HUMAN_REVIEW_PREP = True
RUN_MANUAL_PROJECTION_APPLY = False
RUN_MANUAL_REFERENCE_EVALUATION = False

_REQUIREMENTS_BY_INDEX = {
    int(item["cq_index"]): item for item in read_jsonl(REQUIREMENTS_PATH)}
_SCHEMA_INVENTORY = None


def schema_inventory():
    global _SCHEMA_INVENTORY
    if _SCHEMA_INVENTORY is None:
        _SCHEMA_INVENTORY = load_inventory()
    return _SCHEMA_INVENTORY


# ============================================================
# Prefix normalization wrapper
# ============================================================
# Query extraction is handled by extract_strict_query(...), imported from
# clean_frozen_query_artifacts.py. Prefix repair is centralized there too:
# schema_ttl_prefixes(...) loads the KG-specific schema prefixes, including
# audited aliases, and add_schema_missing_prefixes(...) adds only prefixes that
# are actually used by the query body.

def normalize_query_for_graph(query: str, graph_id: str):
    """Add only used, KG-declared prefixes; preserve the query graph pattern."""
    kg_id = KG_ID_BY_GRAPH[graph_id]
    return add_schema_missing_prefixes(query, schema_ttl_prefixes(kg_id))


def infer_graph_id(input_file):
    path_text = str(input_file).lower()
    name = os.path.basename(path_text)
    if "mcuwiki" in path_text or re.search(r"(?:^|_)small(?:_|\.)", name):
        return "Small"
    if "marveldb" in path_text or re.search(r"(?:^|_)big(?:_|\.)", name):
        return "Big"
    raise ValueError(
        f"Could not infer KG size from input file path: {input_file}")


def perform_eval(input_file):
    graph_id = infer_graph_id(input_file)
    kg_id = KG_ID_BY_GRAPH[graph_id]
    END_POINT = END_POINT_DICT[graph_id]
    evaluator = SingleGraphCQEvaluator(
        endpoint=END_POINT, model_id=TUPLE_MODEL_ID)

    # ============================================================
    # 3. Main processing
    # ============================================================
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_file = f"temp_eval_{base_name}.xlsx"

    # -------------------------
    # 3.1 Read input spreadsheet
    #     (resume from temp if exists)
    # -------------------------
    if os.path.exists(temp_file):
        print(f"Resuming from temp file: {temp_file}")
        df = pd.read_excel(temp_file)
    else:
        print(f"Starting new eval from original file: {input_file}")
        input_xl = pd.ExcelFile(input_file)
        input_sheet = "All" if "All" in input_xl.sheet_names else input_xl.sheet_names[0]
        print(f"Reading sheet: {input_sheet}")
        df = pd.read_excel(input_file, sheet_name=input_sheet)

    required_base_cols = ["CQ", "Prompt"]
    for col in required_base_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ---------------------------------------------
    # 3.2 Detect model-specific analysis columns
    # ---------------------------------------------
    raw_col = None
    result_col = None
    for col in df.columns:
        lc = col.lower()
        if lc.endswith("_raw"):
            raw_col = col
        elif lc.endswith("_result"):
            result_col = col

    if raw_col is None or result_col is None:
        raise ValueError(
            "Could not locate columns ending with '_Raw' and '_Result'.\n"
            f"Columns available: {list(df.columns)}"
        )

    print("Detected analysis columns:")
    print(f"  RAW:    {raw_col}")
    print(f"  RESULT: {result_col}")

    model_name_suffix = "_Result"
    model_name = result_col[: -len(model_name_suffix)
                            ] if result_col.endswith(model_name_suffix) else "model"
    output_file = f"evaluation_results_{model_name}_{graph_id}.xlsx"

    # ------------------------------------------------
    # 3.3 Initialize / add output columns to dataframe
    # ------------------------------------------------
    default_columns = {
        "cq_id": "",
        "cq_index": "",
        "kg_id": kg_id,
        "kg_name": "MCUWiKi" if graph_id == "Small" else "MarvelDB",
        "model_id": model_name,
        "sparql_query": "",
        "evaluation_query": "",
        "query_cleaning_status": "",
        "query_normalization_status": "",
        "added_schema_prefixes": "[]",
        "query_form": "",
        "artifact_status": "",
        "eval_json": "",
        "syntax_ok": "",
        "satisfiable": "",
        "deterministic": "",
        "rows": "",
        "vars": "",
        "latency_p50_ms": "",
        "latency_p95_ms": "",
        "latency_mean_ms": "",
        "lexical_query_overlap": "",
        "semantic_similarity_to_CQ": "",
        "semantic_soft_coverage_to_CQ": "",
        "tuple_cohesion": "",
        "always_unbound_vars": "[]",
        "variables": "[]",
        "determinism_score": "",
        "satisfiability_binding_score": "",
        "h1_overall": "",
        "evaluation_query_sha256": "",
        "execution_kg_snapshot_id": SNAPSHOT_ID_DICT[graph_id],
        LEXICAL_METRIC: "",
        GLOBAL_SEMANTIC_METRIC: "",
        SOFT_SEMANTIC_METRIC: "",
        "alignment_scoring_status": "",
        "selected_schema_term_count": "",
        "selected_schema_terms": "[]",
        "semantic_requirement_scores": "[]",
    }
    for col, default in default_columns.items():
        if col not in df.columns:
            df[col] = default

    rows_done = 0
    if "eval_json" in df.columns:
        rows_done = (df["eval_json"].astype(str).str.strip() != "").sum()
    print(f"Rows already processed: {rows_done}")

    # ------------------------------------------
    # 3.4 Iterate through rows and run execution evaluation
    # ------------------------------------------
    for idx, row in df.iterrows():
        cq_index = idx + 1
        cq_req = _REQUIREMENTS_BY_INDEX.get(cq_index, {})
        cq_id = cq_req.get("cq_id", f"cq_{cq_index:02d}")
        df.at[idx, "cq_index"] = cq_index
        df.at[idx, "cq_id"] = cq_id
        df.at[idx, "kg_id"] = kg_id
        df.at[idx, "model_id"] = model_name

        # Skip expensive endpoint work if this row was already processed in the temp file.
        current_val = str(df.at[idx, "eval_json"])
        if current_val.strip():
            continue

        cq_text = row["CQ"]
        raw_text = row[raw_col]
        analysis_text = row[result_col]

        # Strict extraction removes think/prose wrappers without repairing the query body.
        sparql_query, cleaning_status = extract_strict_query(analysis_text)
        if not sparql_query:
            sparql_query, cleaning_status = extract_strict_query(raw_text)

        evaluation_query, added_prefixes = normalize_query_for_graph(
            sparql_query, graph_id)
        parse_ok = strict_syntax_ok(evaluation_query)

        df.at[idx, "sparql_query"] = sparql_query
        df.at[idx, "evaluation_query"] = evaluation_query
        df.at[idx, "query_cleaning_status"] = cleaning_status
        df.at[idx, "query_normalization_status"] = "schema_prefixes_added" if added_prefixes else "unchanged"
        df.at[idx, "added_schema_prefixes"] = json.dumps(added_prefixes)
        df.at[idx, "evaluation_query_sha256"] = sha256_text(evaluation_query)
        df.at[idx, "syntax_ok"] = parse_ok
        df.at[idx, "query_form"] = query_form(evaluation_query)

        if not sparql_query.strip():
            df.at[idx, "artifact_status"] = "NO_QUERY" if cleaning_status == "empty_model_output" else "NON_SPARQL_OUTPUT"
            df.at[idx, "eval_json"] = df.at[idx, "artifact_status"]
        elif not parse_ok:
            df.at[idx, "artifact_status"] = "PARSE_FAILURE"
            df.at[idx, "eval_json"] = "PARSE_FAILURE"
        elif not RUN_LEGACY_EVALUATOR:
            df.at[idx, "artifact_status"] = "PENDING_EXECUTION_DISABLED"
            df.at[idx, "eval_json"] = "EXECUTION DISABLED"
        else:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        evaluator.evaluate, cq_text, evaluation_query)
                    result = future.result(timeout=EXECUTION_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                df.at[idx, "artifact_status"] = "EXECUTION_TIMEOUT"
                df.at[idx,
                      "eval_json"] = f"ERROR: Timeout after {EXECUTION_TIMEOUT_SECONDS} seconds"
            except Exception as e:
                df.at[idx, "artifact_status"] = "EXECUTION_FAILURE"
                df.at[idx, "eval_json"] = f"ERROR: {e}"
            else:
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except Exception:
                        df.at[idx, "eval_json"] = result
                        result = {}
                    else:
                        df.at[idx, "eval_json"] = json.dumps(result, indent=2)
                else:
                    df.at[idx, "eval_json"] = json.dumps(result, indent=2)

                df.at[idx, "artifact_status"] = "EXECUTABLE" if result.get(
                    "syntax_ok", parse_ok) else "PARSE_FAILURE"
                df.at[idx, "syntax_ok"] = result.get("syntax_ok", parse_ok)
                df.at[idx, "satisfiable"] = result.get("satisfiable", False)
                df.at[idx, "deterministic"] = result.get(
                    "deterministic", False)
                df.at[idx, "rows"] = result.get("rows", 0)
                df.at[idx, "vars"] = result.get("vars", 0)

                latency = result.get("latency", {}) or {}
                df.at[idx, "latency_p50_ms"] = latency.get("p50_ms")
                df.at[idx, "latency_p95_ms"] = latency.get("p95_ms")
                df.at[idx, "latency_mean_ms"] = latency.get("mean_ms")

                df.at[idx, "lexical_query_overlap"] = result.get(
                    "lexical_query_overlap", 0)
                df.at[idx, "semantic_similarity_to_CQ"] = result.get(
                    "semantic_similarity_to_CQ", 0)
                df.at[idx, "semantic_soft_coverage_to_CQ"] = result.get(
                    "semantic_soft_coverage_to_CQ", 0)
                df.at[idx, "tuple_cohesion"] = result.get("tuple_cohesion", 0)
                df.at[idx, "always_unbound_vars"] = json.dumps(
                    result.get("always_unbound_vars", []))
                df.at[idx, "variables"] = json.dumps(
                    result.get("variables", []))

                det_score = 1.0 if bool(result.get(
                    "deterministic", False)) else 0.0
                unbound = result.get("always_unbound_vars", []) or []
                variables = result.get("variables", []) or []
                sat_score = 0.0 if not bool(result.get(
                    "satisfiable", False)) else 1.0 - (len(unbound) / max(len(variables), 1))
                df.at[idx, "determinism_score"] = det_score
                df.at[idx, "satisfiability_binding_score"] = sat_score
                df.at[idx, "h1_overall"] = (det_score + sat_score) / 2

        if idx % 1 == 0:
            df.to_excel(temp_file, index=False)
            print(f"Temp saved at row {idx} -> {temp_file}")
 #       import torch
 #       import gc
 #       gc.collect()
 #       if torch.cuda.is_available():
 #           torch.cuda.empty_cache()
#            torch.cuda.ipc_collect()

    # ------------------------------------------
    # 3.5 Compute paper-facing alignment metrics in batch
    # ------------------------------------------
    if RUN_ALIGNMENT_METRICS:
        records = []
        inventory = schema_inventory()
        for idx, row in df.iterrows():
            cq_index = int(row.get("cq_index") or idx + 1)
            cq_req = _REQUIREMENTS_BY_INDEX.get(cq_index, {})
            evaluated_query = str(row.get("evaluation_query")
                                  or row.get("sparql_query") or "")
            extracted = extract_selected_schema_terms(
                evaluated_query, kg_id, inventory)
            labels = [item["label"] for item in extracted["selected_terms"]]
            cq_tokens = normalize_tokens(str(row["CQ"]))
            label_tokens = normalize_tokens(" ".join(labels))
            eligible = extracted["status"] == "eligible_safe_schema_terms"
            records.append({
                "cq_text": str(row["CQ"]),
                "requirement_units": requirement_units(cq_req) if cq_req else [],
                "safe_schema_term_eligible": eligible,
                "selected_schema_terms": extracted["selected_terms"],
                LEXICAL_METRIC: jaccard(cq_tokens, label_tokens) if eligible else None,
                GLOBAL_SEMANTIC_METRIC: None,
                SOFT_SEMANTIC_METRIC: None,
                "semantic_requirement_scores": [],
                "alignment_scoring_status": extracted["status"],
                "selected_schema_term_count": len(extracted["selected_terms"]),
            })

        add_semantic_scores(
            records,
            model_id=SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_REVISION,
            instruction=INSTRUCTION,
            device=SEMANTIC_DEVICE,
            batch_size=SEMANTIC_BATCH_SIZE,
            cache_dir=SEMANTIC_CACHE_DIR,
        )

        for idx, record in enumerate(records):
            df.at[idx, LEXICAL_METRIC] = record.get(LEXICAL_METRIC)
            df.at[idx, GLOBAL_SEMANTIC_METRIC] = record.get(
                GLOBAL_SEMANTIC_METRIC)
            df.at[idx, SOFT_SEMANTIC_METRIC] = record.get(SOFT_SEMANTIC_METRIC)
            df.at[idx, "alignment_scoring_status"] = record.get(
                "alignment_scoring_status")
            df.at[idx, "selected_schema_term_count"] = record.get(
                "selected_schema_term_count")
            df.at[idx, "selected_schema_terms"] = json.dumps(
                record.get("selected_schema_terms", []), ensure_ascii=True)
            df.at[idx, "semantic_requirement_scores"] = json.dumps(
                record.get("semantic_requirement_scores", []), ensure_ascii=True)

    # ------------------------------------------
    # 3.6 Save spreadsheet with model in name
    # ------------------------------------------
    df.to_excel(output_file, index=False)
    print(f"Saved: {output_file}")

    if EXPORT_RUBRIC_WORKBOOK:
        add_hyp_scores(output_file)

    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Removed temp: {temp_file}")


def get_files_with_extension(directory, extension, recursive=True):
    root = Path(directory)
    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    return [str(path) for path in root.glob(pattern) if path.is_file()]


# Repo-local generated-output workbooks from the original experiment.
directory = PROJECT_ROOT / "evaluation_results"
extension = ".xlsx"
files = get_files_with_extension(directory, extension)
files = [
    file for file in files
    if "evaluation_results_" in Path(file).name
    and not Path(file).name.startswith("~$")
    and not Path(file).name.endswith("_with_rubric.xlsx")
    and "phi4" not in Path(file).name.lower()
]
print(f"Discovered {len(files)} reconstructed non-Phi4 input workbooks")
print(*files, sep="\n")

# Run evaluation for selected files. Keep FILTER_TEXT empty to process all non-Phi4 models.
# SKIP_TEXT = ["mistral-small3.2_latest_Analysis_Big", "llama3.2latest_Analysis_Big", "granite3.3latest_Analysis_Big", "granite4_latest_Analysis_Big", "gemma3_27b_Analysis_Big", "deepseek-r1latest_Analysis_Big", "evaluation_results_GPT5_Big", "Claude_Big"]  # e.g., "GPT5", "Claude", "gemma", or "" for all files
SKIP_TEXT = []  # e.g., "GPT5", "Claude", "gemma", or "" for all files
DONT_SKIP = []

new_files = [file for file in files if any(
    item in Path(file).name for item in DONT_SKIP)]
for file in sorted(new_files, reverse=True):
    if SKIP_TEXT and any(item in Path(file).name for item in SKIP_TEXT):
        continue
    print(file)
    perform_eval(file)
#    import torch
#    import gc
#    gc.collect()
#    if torch.cuda.is_available():
#        torch.cuda.empty_cache()
#        torch.cuda.ipc_collect()
