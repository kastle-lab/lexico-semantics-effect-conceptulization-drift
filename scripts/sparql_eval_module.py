# ----------------------------USAGE-------------------------------------
# evaluator = SingleGraphCQEvaluator(endpoint=END_POINT)

# (cq_text, sparql_query)
# result = evaluator.evaluate(cq_text,sparql_query)

# Response:
# {'cq_text': 'What are all the available modules and their levels',
#  'syntax_ok': True,
#  'satisfiable': True,
#  'deterministic': True,
#  'latency': {'p50_ms': 20.14, 'p95_ms': 26.99, 'mean_ms': 22.22},
#  'rows': 45,
#  'vars': 4,
#  'always_unbound_vars': [],
#  'lexical_query_overlap': 0.0,
#  'semantic_similarity_to_CQ': 0.67,
#  'semantic_soft_coverage_to_CQ': 0.74,
#  'tuple_cohesion': 0.62,
#  'variables': [{'var': 'module',
#    'rows': 45,
#    'distinct': 45,
#    'null_rate': 0.0,
#    'label_resolution_rate': 1.0,
#    'diversity_entropy_norm': 1.0,
#    'semantic_cohesion': 0.56,
#    'dup_ratio': 0.0,
#    'bound_rate': 1.0},
#   {'var': 'moduleTitle',
#    'rows': 45,
#    'distinct': 45,
#    'null_rate': 0.0,
#    'label_resolution_rate': 1.0,
#    'diversity_entropy_norm': 1.0,
#    'semantic_cohesion': 0.56,
#    'dup_ratio': 0.0,
#    'bound_rate': 1.0}]}
# ----------------------------------------------------------------------


import re, time, math, hashlib, statistics, argparse, os
from typing import Dict, List, Tuple, Optional, Callable
from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------------------------------------------------------------
# Internal helpers (used by metrics)
# ---------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _run(endpoint: str, query: str, headers: Optional[Dict[str,str]]=None):
    """
    Execute a SPARQL query on a given endpoint.

    Parameters
    ----------
    endpoint : str
        SPARQL endpoint URL.
    query : str
        A SPARQL query string (SELECT/ASK/CONSTRUCT/DESCRIBE).
    headers : Optional[Dict[str,str]]
        Extra HTTP headers (e.g., to toggle reasoning modes if your store supports it).

    Returns
    -------
    (bindings, raw) : (List[Dict], Dict)
        bindings : list of row dicts in SPARQL JSON format (for SELECT).
        raw      : the full parsed JSON result from the endpoint. For ASK queries,
                   many engines return {'boolean': True/False} at the top level.

    Notes
    -----
    - This helper normalizes responses to the part we need most often (bindings).
    - For ASK queries, you will typically read `raw.get("boolean")`.
    """
    s = SPARQLWrapper(endpoint)
    s.setTimeout(_env_int("SPARQL_EVAL_TIMEOUT_SECONDS", 600))
    s.setReturnFormat(JSON)
    s.setQuery(query)
    if headers:
        for k,v in headers.items(): s.addCustomHttpHeader(k,v)
    res = s.query().convert()

    return res.get("results", {}).get("bindings", []), res


def _normalize_binding(b):
    """
    Normalize one row (binding dict) into simple comparable strings.

    Purpose
    -------
    - Make rows comparable across runs by folding language tags and datatypes into the string.
    - Used by checksums and set-similarity.

    Returns
    -------
    dict : { var_name -> normalized_string }
           where literal values include "@lang" and/or "^^<datatype>" suffixes.
    """
    out = {}
    for k, v in b.items():
        s = v["value"]
        if "xml:lang" in v: s += f"@{v['xml:lang']}"
        if "datatype" in v: s += f"^^<{v['datatype']}>"
        out[k] = s
    return out


def _result_checksum(bindings: List[Dict]) -> str:
    """
    Order-insensitive checksum of a result set.

    Interpretation
    --------------
    - Identical checksums across repeated runs → same multiset of rows (stable results).
    - Useful for determinism testing.

    Returns
    -------
    sha256 hex string for the normalized, sorted representation of rows.
    """
    norm = [tuple(sorted(_normalize_binding(b).items())) for b in bindings]
    norm.sort()
    return hashlib.sha256(repr(norm).encode("utf-8")).hexdigest()


def _jaccard(a, b) -> float:
    """
    Jaccard similarity of two result sets (0..1).

    Interpretation
    --------------
    - 1.0 → exactly identical sets (or both empty).
    - 0.0 → no overlap at all.
    - Useful when comparing results across snapshots or regimes.

    Note
    ----
    - This is unused in the minimal API below but kept for extensibility.
    """
    A = {tuple(sorted(_normalize_binding(x).items())) for x in a}
    B = {tuple(sorted(_normalize_binding(x).items())) for x in b}
    return 1.0 if not (A or B) else len(A & B) / len(A | B)


def _extract_where(q: str) -> Optional[str]:
    """
    Extract the top-level WHERE {...} body using naive brace matching.

    Returns
    -------
    str or None
        The substring inside the OUTERMOST WHERE braces, or None if not found.

    Caveats
    -------
    - This is a pragmatic extractor for typical SELECT queries.
    - It will not handle very exotic formatting or multiple top-level groups.
    """
    
    stack, start, end = 0, None, None
    for i,ch in enumerate(q):
        if ch == "{":
            if stack==0 and start is None: start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack==0: end = i
    return q[start+1:end] if start is not None and end is not None else None

def _extract_prefix_block(q: str) -> str:
    """
    Capture PREFIX and BASE declarations from the query (anywhere).
    Returns them joined with newlines so they can be reused in derived queries (e.g., ASK).
    """
    # Grab full lines starting with PREFIX/BASE (case-insensitive)
    lines = re.findall(r"(?im)^\s*(?:PREFIX|BASE)\s+[^\r\n]+", q)
    return "\n".join(lines) + ("\n" if lines else "")

def _with_probe_limit(query: str, limit: int) -> str:
    if re.search(r"(?is)\bLIMIT\s+\d+\b", query):
        return query
    return query.rstrip() + f"\nLIMIT {limit}"


def disconnected_required_join_risk(query: str) -> bool:
    from rdflib.term import Variable
    from rdflib.plugins.sparql.parser import parseQuery
    from rdflib.plugins.sparql.algebra import translateQuery
    from rdflib.plugins.sparql.parserutils import CompValue

    try:
        algebra = translateQuery(parseQuery(query)).algebra
    except Exception:
        return False

    triples = []

    def walk(value):
        if isinstance(value, CompValue):
            # Conservative: do not fail queries because of OPTIONAL/UNION structure.
            if value.name in {"LeftJoin", "Union"}:
                return
            if value.name == "BGP":
                triples.extend(value.get("triples", []))
                return
            for nested in value.values():
                walk(nested)
        elif isinstance(value, dict):
            for nested in value.values():
                walk(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                walk(nested)

    walk(algebra)

    components = []
    for triple in triples:
        vars_in_triple = {str(term) for term in triple if isinstance(term, Variable)}
        if vars_in_triple:
            components.append(vars_in_triple)

    if len(components) <= 1:
        return False

    merged = []
    for component in components:
        matches = [i for i, existing in enumerate(merged) if existing & component]
        if not matches:
            merged.append(set(component))
            continue
        target = matches[0]
        merged[target].update(component)
        for i in reversed(matches[1:]):
            merged[target].update(merged.pop(i))

    return len(merged) > 1

# ---------------------------------------------------------------------
# 1) Syntax & lint metrics
# ---------------------------------------------------------------------

def syntax_valid(query: str) -> Tuple[bool,str]:
    """
    Quick syntax sanity check (lightweight).

    WHAT
    ----
    - Checks for a non-empty query, balanced braces, and presence of a SPARQL form keyword.

    RETURNS
    -------
    (ok, message) : (bool, str)
        ok       : True if the basic checks pass; False otherwise.
        message  : "OK" on success or a human-readable reason on failure.

    HOW TO INTERPRET
    ----------------
    - True,"OK"     → passes minimal structural sanity checks.
    - False,reason  → likely a malformed or incomplete query string.

    NOTE
    ----
    - For strict SPARQL 1.1 parsing, use rdflib/jena parsers. This function
      aims to be fast and dependency-light.
    """
    if not query.strip(): return False,"Empty query"
    if query.count("{") != query.count("}"): return False,"Unbalanced braces"
    if not re.search(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", query, re.I):
        return False,"No SPARQL form found"
    return True,"OK"


# Common lints that catch portability and determinism pitfalls.
ANTI_PATTERNS = [
    # Pagination should be ordered; OFFSET without ORDER BY can reorder nondeterministically.
    (r"\bOFFSET\s+\d+\b(?![\s\S]*\bORDER\s+BY\b)", "OFFSET without ORDER BY"),
    # Explicit SELECT list improves readability and prevents accidental wide projections.
    (r"\bSELECT\s+\*\b", "SELECT * (prefer explicit projection)"),
    # FILTER(!BOUND(?x)) is often misused to simulate NOT EXISTS; can make OPTIONAL effectively mandatory.
    (r"FILTER\s*\(\s*!?\s*BOUND\s*\(\?\w+\)\s*\)", "FILTER(!BOUND) at tail"),
    # Vendor-specific functions harm portability across triple stores.
    (r"\b(bif:|sql:|pragma:)\w+", "Vendor-specific function detected"),
]

def lint_query(query: str) -> List[str]:
    """
    Lint for common anti-patterns and unused prefixes.

    WHAT
    ----
    - Scans the query text for patterns that often cause portability/logic issues.
    - Flags unused PREFIX declarations.

    RETURNS
    -------
    issues : List[str]
        Human-readable warnings/errors. Empty list means "no issues found".

    HOW TO INTERPRET
    ----------------
    - Longer lists indicate more potential maintainability/portability problems.
    - Not all lints are fatal; treat them as prompts to review the query.
    """
    issues=[]
    for pat,msg in ANTI_PATTERNS:
        if re.search(pat,query,re.I|re.M): issues.append(msg)
    declared = set(re.findall(r"PREFIX\s+(\w+):", query, re.I))
    used = set(re.findall(r"(\w+):\w+", query))
    for p in declared - used:
        issues.append(f"Unused PREFIX: {p}:")
    return issues


def complexity_stats(query: str) -> Dict[str,int]:
    """
    Very rough structural complexity metrics.

    RETURNS
    -------
    dict with:
      - triple_patterns : int    # heuristic count of triple-like patterns
      - optional_blocks : int    # number of OPTIONAL occurrences
      - union_blocks    : int    # number of UNION occurrences
      - subqueries      : int    # number of nested SELECTs (approx)
      - property_paths  : int    # occurrences of /, //, +, * in paths (approx)

    HOW TO INTERPRET
    ----------------
    - Higher numbers generally indicate greater cognitive/computational complexity.
    - Useful as maintainability proxies and for stratifying queries by difficulty.
    """
    return {
        "triple_patterns": len(re.findall(r"\w+\s+\w+[:\w/<>=\-\.\?]+", query)),
        "optional_blocks": len(re.findall(r"\bOPTIONAL\b", query, re.I)),
        "union_blocks": len(re.findall(r"\bUNION\b", query, re.I)),
        "subqueries": max(0, len(re.findall(r"\bSELECT\b", query, re.I))-1),
        "property_paths": len(re.findall(r"[/?][+*]{1,2}", query)),
    }


# ---------------------------------------------------------------------
# 2) Logic checks (reference-free)
# ---------------------------------------------------------------------

def satisfiable(endpoint: str, query: str) -> bool:
    """
    Checks whether the WHERE body can match at least one solution on the dataset.

    WHAT
    ----
    - Converts the main WHERE block to an ASK query: if ASK returns True,
      the query is "satisfiable" on the current data (not necessarily that
      the SELECT will return rows after projection/filters).

    RETURNS
    -------
    bool
        True  → At least one solution exists for the core pattern.
        False → The core pattern/filters likely make it impossible on current data.

    HOW TO INTERPRET
    ----------------
    - False can indicate over-constraining patterns, bad joins, wrong IRIs, or a too-restrictive FILTER.
    - True does NOT guarantee the final SELECT returns non-empty (e.g., projection might drop all vars).
    """
    
    body = _extract_where(query)
    if not body: return False
    prefixes = _extract_prefix_block(query)
    ask = f"{prefixes}ASK WHERE {{ {body} }}"
    _, raw = _run(endpoint, ask)
    return raw.get("boolean", False)


def deterministic(endpoint: str, query: str, runs:int=3) -> bool:
    """
    Re-run the same query multiple times and check exact result-set stability.

    WHAT
    ----
    - Compares order-insensitive checksums of the entire result set across runs.

    RETURNS
    -------
    bool
        True  → Identical results across runs (good sign of determinism).
        False → Results differ (could be due to missing ORDER BY with OFFSET,
                nondeterministic engine behavior, or non-stable data).

    HOW TO INTERPRET
    ----------------
    - Prefer True for production queries users will page through or cache.
    - If False, review ORDER BY / pagination strategy and data volatility.
    """
    checksums=[_result_checksum(_run(endpoint,query)[0]) for _ in range(runs)]
    return len(set(checksums))==1


def mutation_sensitivity(endpoint: str, query: str) -> float:
    """
    Apply small syntactic/semantic mutations and see if results change.

    WHAT
    ----
    - Mutations simulate common logic errors (remove DISTINCT, flip = to !=, strip a FILTER).
    - We count how many mutations change the result vs the original.

    RETURNS
    -------
    ratio : float in [0,1]
        = changed / tested
        - 1.0 → Every mutation altered behavior (your query/tests are discriminative).
        - 0.0 → Mutations had no effect (the query may be too weak or mutations irrelevant).

    HOW TO INTERPRET
    ----------------
    - Higher ratio is generally better: the query is "meaningful" and not trivially insensitive.
    - Low ratio can indicate that important constraints are missing (e.g., DISTINCT not needed,
      FILTERs unused, or joins not selective).
    """
    mutations=[
        # Remove DISTINCT → should often increase duplicates (change results).
        lambda q: re.sub(r"\bDISTINCT\b","",q,flags=re.I),
        # Flip first equality to inequality → should change rows if the condition is effective.
        lambda q: re.sub(r"=\s*([^\s)]+)",r"!= \1",q,1),
        # Remove first FILTER → results may broaden if filter was effective.
        lambda q: re.sub(r"\bFILTER\s*\([^()]+\)","",q,1,flags=re.I),
    ]
    base_cs=_result_checksum(_run(endpoint,query)[0])
    tested,changed=0,0
    for m in mutations:
        mq=m(query)
        if mq!=query:
            tested+=1
            try:
                if _result_checksum(_run(endpoint,mq)[0])!=base_cs:
                    changed+=1
            except Exception:
                # If the mutation becomes invalid/throws, we treat as changed behavior.
                changed+=1
    return changed/tested if tested else 0.0


# ---------------------------------------------------------------------
# 3) Performance metrics
# ---------------------------------------------------------------------

def latency(endpoint: str, query: str, repeats:int=5) -> Dict[str,float]:
    """
    Measure wall-clock execution time across multiple runs.

    WHAT
    ----
    - Executes the query `repeats` times and aggregates basic latency percentiles.

    RETURNS
    -------
    dict with (milliseconds):
      - p50_ms : float  # median latency (typical)
      - p95_ms : float  # tail latency (slow outliers)
      - p99_ms : float  # extreme tail
      - mean_ms: float  # average latency

    HOW TO INTERPRET
    ----------------
    - p50_ms ≈ everyday experience; p95_ms is critical for SLOs.
    - Compare across queries/datasets or cold vs warm cache scenarios.
    """
    times=[]
    for _ in range(repeats):
        t0=time.time(); _run(endpoint,query); times.append((time.time()-t0)*1000)
    times.sort()
    pct=lambda p: times[min(len(times)-1, math.ceil(p*len(times))-1)]
    return {
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "mean_ms": statistics.mean(times)
    }


# ---------------------------------------------------------------------
# 4) Maintainability proxies
# ---------------------------------------------------------------------

def variable_hygiene(query: str) -> Dict[str,object]:
    """
    Simple variable usage hygiene statistics.

    RETURNS
    -------
    dict with:
      - all_vars      : List[str]  # all variable names found (e.g., ['s','x'])
      - unused_vars   : List[str]  # variables that appear outside WHERE or not truly used
      - name_entropy  : float      # avg(#unique characters) across variable names

    HOW TO INTERPRET
    ----------------
    - Many `unused_vars` can indicate dead code or accidental cross-joins.
    - Higher `name_entropy` suggests more informative names (very rough proxy).
    - These are heuristics; use alongside lints/complexity for maintainability.
    """
    vars_all=set(re.findall(r"\?([A-Za-z_]\w+)",query))
    body=_extract_where(query) or ""
    where_use=set(re.findall(r"\?([A-Za-z_]\w+)",body))
    return {
        "all_vars":sorted(vars_all),
        "unused_vars":sorted(vars_all-where_use),
        "name_entropy":statistics.mean([len(set(v)) for v in vars_all]) if vars_all else 0.0
    }

# ---------------------------------------------------------------------
# CLI demo: run all metrics on a query and return a dict
# ---------------------------------------------------------------------

def run_eval(endpoint: str, query: str):
    """
    Convenience wrapper that runs all metrics and prints/returns the results.

    RETURNS
    -------
    dict
      {
        "syntax_valid":          (bool,str),      # minimal syntax sanity
        "lint_query":            List[str],       # list of warnings/errors (empty = clean)
        "complexity_stats":      Dict[str,int],   # structural counts
        "satisfiable":           bool,            # ASK-converted satisfiability
        "deterministic":         bool,            # stable results across runs
        "mutation_sensitivity":  float,           # [0,1] higher = more discriminative
        "latency":               Dict[str,float], # p50/p95/p99/mean in ms
        "variable_hygiene":      Dict[str,object] # var usage & name entropy
      }

    HOW TO INTERPRET (quick guide)
    ------------------------------
    - syntax_valid: True/"OK" is expected. False indicates a malformed query.
    - lint_query:   Aim for []. Address findings to improve portability/clarity.
    - complexity_stats: Higher counts → more complex; compare relative to peers.
    - satisfiable:  False often means over-constrained or wrong graph patterns.
    - deterministic: True is preferred for predictable pagination/caching.
    - mutation_sensitivity: Aim higher (e.g., ≥0.5) to avoid "toothless" queries.
    - latency:      Watch p95_ms against your SLOs.
    - variable_hygiene: Fewer `unused_vars` is better; moderate/high `name_entropy` is nice.
    """
    print("Syntax:", syntax_valid(query))
    print("Lint:", lint_query(query))
    print("Complexity:", complexity_stats(query))
    print("Satisfiable:", satisfiable(endpoint, query))
    print("Deterministic:", deterministic(endpoint, query))
    print("Mutation sensitivity:", mutation_sensitivity(endpoint, query))
    print("Latency:", latency(endpoint, query))
    print("Variable hygiene:", variable_hygiene(query))
    return {
        "syntax_valid": syntax_valid(query),
        "lint_query": lint_query(query),
        "complexity_stats": complexity_stats(query),
        "satisfiable": satisfiable(endpoint, query),
        "deterministic": deterministic(endpoint, query),
        "mutation_sensitivity": mutation_sensitivity(endpoint, query),
        "latency": latency(endpoint, query),
        "variable_hygiene": variable_hygiene(query)
    }


# ======================================================================
# Single-Graph CQ Evaluator — evaluate(cq_text, sparql)
# Transformers embeddings via Sentence-Transformers (auto-downloads model)
# No hardcoded prefixes; dynamic label discovery; semantic NL↔Query scores.
# Depends on your existing helpers: _run, _extract_where, syntax_valid,
# satisfiable, deterministic, latency
# ======================================================================

import os, re, math
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np

# ---------------- basics ----------------
_RE_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_RE_NONALNUM = re.compile(r"[^0-9a-zA-Z@^<>:/# ]+")

def _norm_text(s: str) -> str:
    t = _RE_CAMEL.sub(" ", str(s)).replace("_", " ").replace("-", " ")
    t = _RE_NONALNUM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _dup_ratio(values: List[str]) -> float:
    n = len(values)
    return 0.0 if n <= 1 else max(0.0, 1.0 - (len(set(values)) / n))

def _entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0: return 0.0
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log(p + 1e-12, 2)
    return H

def _lex_overlap(a: str, b: str) -> float:
    A = set(_norm_text(a).split())
    B = set(_norm_text(b).split())
    return 1.0 if not (A or B) else len(A & B) / len(A | B)

def _is_uri(v: dict) -> bool:
    return v.get("type") == "uri"

def _tail(u: str) -> str:
    u = str(u)
    if "#" in u: u = u.rsplit("#", 1)[-1]
    if "/" in u: u = u.rsplit("/", 1)[-1]
    return u

def _project_vars(query: str) -> List[str]:
    m = re.search(r"(?is)SELECT\s+(DISTINCT\s+)?(.+?)\s+WHERE\s*\{", query)
    if not m:
        body = _extract_where(query) or ""
        return sorted(set(re.findall(r"\?([A-Za-z_]\w+)", body)))
    proj = m.group(2)
    if "*" in proj:
        body = _extract_where(query) or ""
        return sorted(set(re.findall(r"\?([A-Za-z_]\w+)", body)))
    return re.findall(r"\?([A-Za-z_]\w+)", proj)

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# ======== SEMANTIC NL↔QUERY MATCH (uses evaluator's embedder) ========

def _localnames_from_query(query_text: str) -> List[str]:
    locs: List[str] = []
    # IRI tails
    for iri in re.findall(r"<([^>]+)>", query_text):
        tail = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        locs.append(_norm_text(tail))
    # QNames
    for _, local in re.findall(r"\b([A-Za-z_][\w\-]*):([A-Za-z_][\w\-]*)\b", query_text):
        locs.append(_norm_text(local))
    # dedupe, drop empties
    return list(dict.fromkeys([t for t in locs if t]))

def _vars_from_query(query_text: str) -> List[str]:
    vs = [m for m in re.findall(r"\?([A-Za-z_]\w+)", query_text)]
    return list(dict.fromkeys([_norm_text(v) for v in vs]))

def _content_tokens(text: str) -> List[str]:
    toks = [t for t in re.split(r"\s+", _norm_text(text)) if t]
    return [t for t in toks if len(t) > 2] or toks

# --------------- embeddings: Sentence-Transformers (auto-download) ---------------
class _EmbedderBase:
    def embed(self, texts: List[str]) -> np.ndarray: raise NotImplementedError

class _SBERT(_EmbedderBase):
    """
    Uses Sentence-Transformers. Auto-downloads the model from Hugging Face
    the first time it’s needed (cached afterwards).
    Set SBERT_MODEL to override the model id/path.
      e.g., SBERT_MODEL=sentence-transformers/all-MiniLM-L6-v2
    """
    def __init__(self, model_id: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        # Pick model from env or a strong, lightweight default.
        self.model_id = (model_id or os.environ.get("SBERT_MODEL") 
                         or "sentence-transformers/all-MiniLM-L6-v2")
        # This will download automatically if not present locally.
        self.m = SentenceTransformer(self.model_id)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts: 
            # dimension will be inferred on first non-empty call; 1-col zero is OK for empties
            return np.zeros((0, 1), dtype=np.float32)
        #vecs = self.m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        batch_size = _env_int("SPARQL_EVAL_EMBED_BATCH_SIZE", 64)
        vecs = self.m.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                )
        return np.array(vecs, dtype=np.float32)

class _LocalTinyHash(_EmbedderBase):
    """Dependency-free deterministic embedder for local flow tests only."""
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in _content_tokens(str(text)):
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dimensions
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            rows.append(vector)
        if not rows:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack(rows).astype(np.float32, copy=False)

def _pick_embedder() -> _EmbedderBase:
    # Single, explicit strategy: use Sentence-Transformers and let it fetch the model.
    # No local search heuristics; model id can be overridden via SBERT_MODEL.
    return _SBERT()

def _embed_same_space(embedder: _EmbedderBase, groups: List[List[str]]) -> List[np.ndarray]:
    """
    With SBERT (fixed-size embeddings), each group is embedded independently.
    This ensures empty groups are returned with the correct dimensionality.
    """
    outs=[]
    max_dim=None
    # first pass: embed non-empty groups to learn dim
    for g in groups:
        if g:
            X=embedder.embed(g)
            outs.append(X)
            max_dim = X.shape[1] if max_dim is None else max(max_dim, X.shape[1])
        else:
            outs.append(None)
    if max_dim is None:
        # all empty
        return [np.zeros((0,1), dtype=np.float32) for _ in groups]
    # second pass: fill empties with zeros of learned dim
    final=[]
    for X in outs:
        if X is None:
            final.append(np.zeros((0,max_dim), dtype=np.float32))
        elif X.shape[1]==max_dim:
            final.append(X)
        else:
            # very unlikely mismatch; pad/truncate to max_dim defensively
            pad=np.zeros((X.shape[0], max_dim), dtype=np.float32)
            pad[:, :min(X.shape[1], max_dim)] = X[:, :min(X.shape[1], max_dim)]
            final.append(pad)
    return final

def nl_query_semantic_scores(cq_text: str, sparql_query: str, embedder: _EmbedderBase) -> dict:
    """
    Two semantic scores in [0,1], using a shared embedding space:
      - semantic_similarity_to_CQ: CQ sentence vs mean of query terms
      - semantic_soft_coverage_to_CQ: avg over CQ tokens of max sim to any query term
    """
    where_body = _extract_where(sparql_query) or ""
    query_terms = (
        _vars_from_query(sparql_query)
        + _localnames_from_query(sparql_query)
        + _content_tokens(where_body)
    )
    query_terms = list(dict.fromkeys([t for t in query_terms if t]))
    cq_toks = _content_tokens(cq_text)

    if not query_terms:
        return {"semantic_similarity_to_CQ": 0.0, "semantic_soft_coverage_to_CQ": 0.0}

    E_terms, E_cq, E_tok = _embed_same_space(embedder, [query_terms, [cq_text], cq_toks])

    if E_terms.shape[0] == 0:
        return {"semantic_similarity_to_CQ": 0.0, "semantic_soft_coverage_to_CQ": 0.0}

    # centroid of query terms
    E_terms_mean = E_terms.mean(axis=0, keepdims=True)
    E_terms_mean /= (np.linalg.norm(E_terms_mean, axis=1, keepdims=True) + 1e-8)

    # whole-sentence similarity
    if E_cq.shape[0] == 0:
        sim01 = 0.0
    else:
        sim = float(np.clip((E_cq @ E_terms_mean.T).squeeze(), -1.0, 1.0))
        sim01 = 0.5 * (sim + 1.0)

    # token soft coverage
    if E_tok.shape[0] == 0:
        cov01 = sim01
    else:
        S = E_tok @ E_terms.T
        best = S.max(axis=1)
        cov01 = float(np.clip(np.mean(0.5 * (best + 1.0)), 0.0, 1.0))

    return {
        "semantic_similarity_to_CQ": round(sim01,2) if sim01 else sim01,
        "semantic_soft_coverage_to_CQ": round(cov01,2) if cov01 else cov01
    }

def _set_cohesion(texts: List[str], embedder: _EmbedderBase) -> float:
    uniq = list({ _norm_text(t) for t in texts if str(t).strip() })
    max_values = _env_int("SPARQL_EVAL_MAX_COHESION_VALUES", 0)
    if max_values > 0:
        uniq = uniq[:max_values]
    if len(uniq) <= 1: return 1.0
    emb = embedder.embed(uniq)
    if emb.shape[0] <= 1: return 1.0
    S = _cosine(emb, emb)
    np.fill_diagonal(S, -1.0)
    return float(S.max(axis=1).mean())

# --------- dynamic label discovery (schema-agnostic) ---------
def _labels_for_uris(endpoint: str, uris: List[str], headers: Optional[Dict[str,str]]=None) -> Dict[str,str]:
    """
    Discover a label-like literal per URI with no prefix/property assumptions.
    """
    uri2label: Dict[str,str] = {}
    if not uris: return uri2label
    max_uris = _env_int("SPARQL_EVAL_MAX_LABEL_URIS", 0)
    if max_uris > 0:
        uris = list(dict.fromkeys(uris))[:max_uris]

    # chunk VALUES size safely (budget proportional to average URI length)
    avg_len = max(16, int(sum(len(u) for u in uris) / max(1, len(uris))))
    budget = avg_len * 200
    chunk, size = [], 0
    chunks: List[List[str]] = []
    for u in sorted(set(uris)):
        tok = f"<{u}>"
        if size + len(tok) + 1 > budget and chunk:
            chunks.append(chunk); chunk = [u]; size = len(tok) + 1
        else:
            chunk.append(u); size += len(tok) + 1
    if chunk: chunks.append(chunk)

    predicate_counts: Counter = Counter()
    per_subj: Dict[str, List[Tuple[str,str,dict]]] = defaultdict(list)

    for group in chunks:
        values = " ".join(f"<{u}>" for u in group)
        q = f"""
SELECT ?s ?p ?o ?lang ?dt WHERE {{
  VALUES ?s {{ {values} }}
  ?s ?p ?o .
  FILTER(isLiteral(?o))
  BIND(LANG(?o) AS ?lang)
  BIND(DATATYPE(?o) AS ?dt)
}}"""
        try:
            rows, _ = _run(endpoint, q, headers=headers)
        except Exception:
            rows = []
        for r in rows:
            s = r["s"]["value"]; p = r["p"]["value"]; o = r["o"]["value"]
            lang = r.get("lang", {}).get("value", "")
            dt = r.get("dt", {}).get("value", "")
            predicate_counts[p] += 1
            per_subj[s].append((p, o, {"lang": lang, "dt": dt}))

    if not predicate_counts:
        return uri2label

    maxc = max(predicate_counts.values())
    pred_score = {p: c/maxc for p,c in predicate_counts.items()}
    all_lengths = [len(o) for lst in per_subj.values() for _,o,_ in lst]
    median_len = float(np.median(all_lengths)) if all_lengths else 20.0

    def lit_score(p: str, o: str, meta: dict) -> float:
        f = pred_score.get(p, 0.0)
        L = 1.0 if meta.get("lang") else 0.0
        ell = len(o)
        closeness = 1.0 - (abs(ell - median_len) / (median_len + 1e-6))
        closeness = float(np.clip(closeness, 0.0, 1.0))
        punct_pen = 1.0 - (len(re.findall(r"[^0-9A-Za-z\s]", o)) / max(1.0, ell))
        return f * max(0.0, L*0.5 + 0.5) * ((closeness + punct_pen) / 2.0)

    for s, cand in per_subj.items():
        if not cand: continue
        best = max(cand, key=lambda t: lit_score(*t))
        uri2label[s] = best[1]
    return uri2label

# ---------------- two-arg public API ----------------
class SingleGraphCQEvaluator:
    """
    Construct with endpoint (and optional headers). Then call:
        evaluate(cq_text, sparql_query)
    - Embeddings: Sentence-Transformers only (auto-downloads; override with SBERT_MODEL).
    - Labels: discovered dynamically; no prefixes/properties hardcoded.
    """
    def __init__(self, endpoint: str, headers: Optional[Dict[str,str]]=None, model_id: Optional[str]=None):
        self.endpoint = endpoint
        self.headers = headers or {}
        # If you want to pin a specific model in code, pass model_id.
        # Otherwise SBERT_MODEL env var or default 'sentence-transformers/all-MiniLM-L6-v2' is used.
        self.embedder = (
            _LocalTinyHash()
            if model_id == "local-tiny-hash"
            else _SBERT(model_id=model_id)
        )

    def evaluate(self, cq_text: str, sparql: str) -> Dict[str, Any]:
        ok, msg = syntax_valid(sparql)
        if not ok:
            return {"cq_text": cq_text, "syntax_ok": False, "syntax_msg": msg}
        
        probe_limit = _env_int("SPARQL_EVAL_CARTESIAN_PROBE_LIMIT", 100000)
        join_risk = disconnected_required_join_risk(sparql)
        
        if join_risk and probe_limit > 0:
            probe_query = _with_probe_limit(sparql, probe_limit)
            probe_rows, _ = _run(self.endpoint, probe_query, headers=self.headers)
            if len(probe_rows) >= probe_limit:
                where_body = _extract_where(sparql) or ""
                vars_ = _project_vars(sparql) or sorted(
                    set(re.findall(r"\?([A-Za-z_]\w+)", where_body))
                )
                return {
                "cq_text": cq_text,
                "syntax_ok": True,
                "artifact_status": "EXECUTION_INVALID_CARTESIAN_JOIN",
                "satisfiable": bool(probe_rows),
                "deterministic": False,
                "latency": {"p50_ms": None, "p95_ms": None, "mean_ms": None},
                "rows": len(probe_rows),
                "rows_lower_bound": probe_limit,
                "vars": len(vars_),
                "always_unbound_vars": [],
                "lexical_query_overlap": (
                    round(_lex_overlap(cq_text, where_body), 2)
                    if _lex_overlap(cq_text, where_body)
                    else _lex_overlap(cq_text, where_body)
                ),
                "tuple_cohesion": None,
                "variables": [],
                "disconnected_join_risk": True,
                "cartesian_probe_limit": probe_limit,
                }

        where_body = _extract_where(sparql) or ""
        sat = satisfiable(self.endpoint, sparql)
        det = deterministic(self.endpoint, sparql, runs=2)
        lat = latency(self.endpoint, sparql, repeats=3)

        bindings, _ = _run(self.endpoint, sparql, headers=self.headers)
        vars_ = _project_vars(sparql) or sorted(set(re.findall(r"\?([A-Za-z_]\w+)", where_body)))

        per_var_vals: Dict[str, List[dict]] = {v: [] for v in vars_}
        uris: List[str] = []
        for r in bindings:
            for v in vars_:
                if v in r:
                    per_var_vals[v].append(r[v])
                    if _is_uri(r[v]): uris.append(r[v]["value"])

        uri2label = _labels_for_uris(self.endpoint, uris, headers=self.headers)

        per_var: List[Dict[str, Any]] = []
        always_unbound: List[str] = []

        for v in vars_:
            vals = per_var_vals[v]
            rows_n = len(bindings)
            bound = len(vals)
            br = (bound / rows_n) if rows_n else 0.0
            if br == 0.0:
                always_unbound.append(v)
                per_var.append({
                    "var": v, "rows": rows_n, "distinct": 0,
                    "null_rate": 1.0, "label_resolution_rate": 0.0,
                    "diversity_entropy_norm": 0.0, "semantic_cohesion": 1.0,
                    "dup_ratio": 0.0, "bound_rate": 0.0
                })
                continue

            # surface strings (dynamic labels)
            texts: List[str] = []
            resolved = 0
            for b in vals:
                raw = b["value"]
                if _is_uri(b):
                    lab = uri2label.get(raw)
                    if lab: texts.append(lab); resolved += 1
                    else:   texts.append(_tail(raw))
                else:
                    s = raw
                    if "xml:lang" in b: s += f"@{b['xml:lang']}"
                    if "datatype" in b: s += f"^^<{b['datatype']}>"
                    texts.append(s)

            norm_texts = [_norm_text(t) for t in texts]
            counts = Counter(norm_texts)
            H = _entropy(counts)
            H_norm = H / math.log(len(counts) + 1e-12, 2) if counts else 0.0
            coh = _set_cohesion(texts, self.embedder)
            dup = _dup_ratio(norm_texts)
            denom_uri = sum(1 for b in vals if _is_uri(b))
            label_rate = (resolved / denom_uri) if denom_uri else 1.0
            nulls = rows_n - bound

            per_var.append({
                "var": v,
                "rows": rows_n,
                "distinct": len(counts),
                "null_rate": (nulls / rows_n) if rows_n else 0.0,
                "label_resolution_rate": label_rate,
                "diversity_entropy_norm": round(H_norm, 2) if H_norm else H_norm,
                "semantic_cohesion": round(coh, 2) if coh else coh,
                "dup_ratio": round(dup, 2) if dup else dup,
                "bound_rate": br
            })

        # Tuple cohesion (uses same embedder)
        tuple_coh: Optional[float] = None
        if bindings and vars_:
            tuples = []
            for r in bindings:
                parts = []
                for v in vars_:
                    if v in r:
                        raw = r[v]["value"]
                        if _is_uri(r[v]):
                            parts.append(_norm_text(uri2label.get(raw, _tail(raw))))
                        else:
                            s = raw
                            if "xml:lang" in r[v]: s += f"@{r[v]['xml:lang']}"
                            if "datatype" in r[v]: s += f"^^<{r[v]['datatype']}>"
                            parts.append(_norm_text(s))
                    else:
                        parts.append("")
                tuples.append(" | ".join(parts))
            uniq = list(dict.fromkeys(tuples))
            max_values = _env_int("SPARQL_EVAL_MAX_COHESION_VALUES", 0)
            if max_values > 0:
                uniq = uniq[:max_values]
            emb = self.embedder.embed(uniq)
            if emb.shape[0] <= 1:
                tuple_coh = 1.0
            else:
                S = _cosine(emb, emb)
                np.fill_diagonal(S, -1.0)
                tuple_coh = float(S.max(axis=1).mean())

        # Semantic NL↔Query alignment (embeddings, not string match)
        sem = nl_query_semantic_scores(cq_text, sparql, self.embedder)

        return {
            "cq_text": cq_text,
            "syntax_ok": True,
            "satisfiable": sat,
            "deterministic": det,
            "latency": {"p50_ms": round(lat["p50_ms"],2), "p95_ms": round(lat["p95_ms"],2), "mean_ms": round(lat["mean_ms"],2)},
            "rows": len(bindings),
            "vars": len(vars_),
            "always_unbound_vars": always_unbound,
            "lexical_query_overlap": round(_lex_overlap(cq_text, where_body),2) if _lex_overlap(cq_text, where_body) else _lex_overlap(cq_text, where_body),  # legacy lexical metric (optional)
            **sem,  # adds semantic_similarity_to_CQ & semantic_soft_coverage_to_CQ
            "tuple_cohesion": round(tuple_coh, 2) if tuple_coh else tuple_coh,
            "variables": per_var,
        }
