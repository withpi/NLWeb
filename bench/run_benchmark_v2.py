#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate local /who API against HF qrels, with optional JSON dump and comparisons.

Examples:
    # Baseline only, save JSON
    python run_benchmark.py \
        --output report.html \
        --dump-json current.json

    # Compare to a prior run
    python run_benchmark.py \
        --output report_compare.html \
        --dump-json current.json \
        --compare-json best=best.json

    # Up to 3 comparisons
    python run_benchmark.py \
        --output report_compare3.html \
        --dump-json current.json \
        --compare-json best=best.json ablation=abl.json alt=alt.json

Requires:
    pip install datasets requests matplotlib numpy pandas tqdm
"""

import argparse
import base64
import importlib
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # for safe non-blocking file rendering
import matplotlib.pyplot as plt


# --- Begin: default URL->corpus_id mapper ------------------------------------
from functools import lru_cache

DEFAULT_DATASET = "withpi/nlweb_who_achint_qrel"
DEFAULT_CORPUS_CONFIG = "corpus"
DEFAULT_SPLIT = "train"

@lru_cache(maxsize=1)
def _url_to_corpus_id_map(
    dataset_name: str = DEFAULT_DATASET,
    corpus_config: str = DEFAULT_CORPUS_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> dict:
    """
    One-time load of the corpus split, building {url -> corpus_id}.
    Cached via lru_cache so subsequent calls are O(1).
    """
    ds_corpus = load_dataset(dataset_name, corpus_config, split=split)
    mapping = {}
    for ex in ds_corpus:
        url = ex.get("url", None)
        cid = ex.get("corpus_id", None)
        if url is None or cid is None:
            continue
        mapping[str(url)] = int(cid)
    return mapping

def default_get_doc_id(schema_object: str) -> int:
    """
    Parse schema_object JSON, read 'url', and return corpus_id.
    If anything fails or not found, return -1.
    """
    try:
        obj = json.loads(schema_object)
        original_url = obj.get("url", None)
        if not original_url:
            print(f"Unknown docid! Returning -1. Object: {schema_object}")
            return -1
        
        # Normalize URL to include protocol prefix. The corpus dataset stores URLs with https://
        # but the API may return bare domains like "who.int/news/item/..." without protocol.
        # Without normalization, lookup fails because "who.int/..." != "https://who.int/..."
        if not original_url.startswith(('http://', 'https://')):
            original_url = f"https://{original_url}"
        
        return _url_to_corpus_id_map().get(original_url, -1)
    except Exception:
        print(f"Unknown docid! Returning -1. Object: {schema_object}")
        return -1
# --- End: default URL->corpus_id mapper --------------------------------------

# --------------------------
# Types and utilities
# --------------------------

@dataclass
class Query:
    query_id: int
    text: str


def load_queries(dataset_name: str, queries_config: str, split: str) -> List[Query]:
    ds_queries = load_dataset(dataset_name, queries_config, split=split)
    queries: List[Query] = []
    for ex in ds_queries:
        qid = int(ex["query_id"])
        text = ex["query_text"]
        queries.append(Query(qid, text))
    return queries


def load_qrels_by_field(
    dataset_name: str,
    qrel_config: str,
    split: str,
    field: str,
) -> Dict[int, set]:
    """
    Load qrels mapping query_id -> set(corpus_ids) using the given field:
      - 'positive_corpus_ids_all'
      - 'positive_corpus_ids_critical_only'
    """
    ds_qrel = load_dataset(dataset_name, qrel_config, split=split)
    qrels: Dict[int, set] = {}
    for ex in ds_qrel:
        qid = int(ex["query_id"])
        pos = ex.get(field, None)
        if pos is None:
            pos = []
        qrels[qid] = set(int(x) for x in pos)
    return qrels


def default_headers() -> Dict[str, str]:
    return {"User-Agent": "IR-Eval/0.1"}


def call_api(
    base_url: str,
    query_text: str,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0,
) -> Dict:
    """Call GET {base_url}?query=...&streaming=false and return parsed JSON."""
    params = {"query": query_text, "streaming": "false"}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(base_url, params=params, headers=default_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"API call failed after {retries} attempts: {last_err}")


def load_docid_func(docid_func_spec: str) -> Callable[[str], int]:
    """
    Load get_doc_id from a spec like 'module:function'.
    The function must accept a single str (schema_object) and return an int corpus_id.
    """
    if ":" not in docid_func_spec:
        raise ValueError("--docid-func must be in the form 'module:function'")
    module_name, func_name = docid_func_spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"{docid_func_spec} is not callable")
    return func


# --------------------------
# Metrics
# --------------------------

def precision_at_k(binary_rels: Sequence[int], k: int) -> float:
    k = min(k, len(binary_rels))
    if k <= 0:
        return 0.0
    return float(np.sum(binary_rels[:k])) / float(k)


def recall_at_k(binary_rels: Sequence[int], k: int, num_relevant: int) -> float:
    if num_relevant == 0:
        return 0.0
    k = min(k, len(binary_rels))
    if k <= 0:
        return 0.0
    return float(np.sum(binary_rels[:k])) / float(num_relevant)


def dcg_at_k(binary_rels: Sequence[int], k: int) -> float:
    k = min(k, len(binary_rels))
    dcg = 0.0
    for i in range(k):
        rel = binary_rels[i]
        if rel:
            dcg += 1.0 / math.log2(i + 2)  # positions are 1-indexed
    return dcg


def idcg_at_k(num_relevant: int, k: int) -> float:
    # For binary relevance, the ideal list is all 1s up to min(num_relevant, k)
    m = min(num_relevant, k)
    return sum(1.0 / math.log2(i + 2) for i in range(m))


def ndcg_at_k(binary_rels: Sequence[int], k: int, num_relevant: int) -> float:
    ideal = idcg_at_k(num_relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(binary_rels, k) / ideal


def average_precision_at_k(binary_rels: Sequence[int], k: int, num_relevant: int) -> float:
    """AP@k with denominator min(num_relevant, k); standard for binary rels at cutoff."""
    denom = max(1, min(num_relevant, k))
    ap_sum = 0.0
    max_i = min(k, len(binary_rels))
    rel_so_far = 0
    for i in range(max_i):
        if binary_rels[i]:
            rel_so_far += 1
            ap_sum += rel_so_far / float(i + 1)
    return ap_sum / float(denom)


def reciprocal_rank_at_k(binary_rels: Sequence[int], k: int) -> float:
    max_i = min(k, len(binary_rels))
    for i in range(max_i):
        if binary_rels[i]:
            return 1.0 / float(i + 1)
    return 0.0


# --------------------------
# Evaluation
# --------------------------

@dataclass
class PerQueryEval:
    query_id: int
    num_relevant: int
    retrieved_doc_ids: List[int]
    binary_rels: List[int]  # aligned with retrieved_doc_ids
    categories: List[str]  # aligned with retrieved_doc_ids - extracted from schema_object
    snippets: List[str]  # aligned with retrieved_doc_ids - text preview of document
    urls: List[str]  # aligned with retrieved_doc_ids - document URLs


def evaluate(
    qid_to_gt: Dict[int, set],
    qid_to_results: Dict[int, List[int]],
    qid_to_categories: Dict[int, List[str]],
    qid_to_snippets: Dict[int, List[str]],
    qid_to_urls: Dict[int, List[str]],
    max_k_for_curve: int = 100,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[PerQueryEval], np.ndarray]:
    """
    Compute:
      - Recall@k curve (k=0..max_k_for_curve) averaged across queries
      - Recall@{1,5,10,20,50,100,500} table (macro-avg)
      - NDCG/MAP/MRR at {10,50,100} table (macro-avg)
      - Dense recall curve as numpy array
    """
    per_query: List[PerQueryEval] = []
    for qid, gt_set in qid_to_gt.items():
        results = qid_to_results.get(qid, [])
        categories = qid_to_categories.get(qid, ["unknown"] * len(results))
        snippets = qid_to_snippets.get(qid, [""] * len(results))
        urls = qid_to_urls.get(qid, [""] * len(results))
        rels = [1 if doc_id in gt_set else 0 for doc_id in results]
        per_query.append(PerQueryEval(qid, len(gt_set), results, rels, categories, snippets, urls))

    # Filter out queries with zero relevant docs (avoid divide-by-zero in recall)
    eval_qs = [pq for pq in per_query if pq.num_relevant > 0]
    if not eval_qs:
        raise RuntimeError("No queries with at least 1 relevant document found in qrels.")

    # Recall@k curve
    ks_curve = np.arange(0, max_k_for_curve + 1, dtype=int)
    dense = []
    for k in ks_curve:
        if k == 0:
            dense.append(0.0)
        else:
            vals = [recall_at_k(pq.binary_rels, k, pq.num_relevant) for pq in eval_qs]
            dense.append(float(np.mean(vals)))
    dense = np.array(dense, dtype=float)

    # Recall table
    recall_ks = [1, 5, 10, 20, 50, 100] #, 500]
    recall_rows = []
    for k in recall_ks:
        vals = [recall_at_k(pq.binary_rels, k, pq.num_relevant) for pq in eval_qs]
        recall_rows.append({"k": k, "Recall": float(np.mean(vals))})
    recall_df = pd.DataFrame(recall_rows)

    # NDCG/MAP/MRR at cutoffs
    cutoffs = [10, 50, 100]
    rows = []
    for k in cutoffs:
        ndcgs = [ndcg_at_k(pq.binary_rels, k, pq.num_relevant) for pq in eval_qs]
        maps = [average_precision_at_k(pq.binary_rels, k, pq.num_relevant) for pq in eval_qs]
        mrrs = [reciprocal_rank_at_k(pq.binary_rels, k) for pq in eval_qs]
        rows.append({
            "k": k,
            "NDCG": float(np.mean(ndcgs)),
            "MAP": float(np.mean(maps)),
            "MRR": float(np.mean(mrrs)),
        })
    metrics_df = pd.DataFrame(rows)

    return ks_curve, recall_df, metrics_df, eval_qs, dense


def plot_recall_curve_multi(
    ks_curve: np.ndarray,
    current_dense: np.ndarray,
    comp_curves: List[Tuple[str, np.ndarray]],
    title: str,
):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(ks_curve, current_dense, linewidth=2, label="current")
    for name, curve in comp_curves:
        m = min(len(ks_curve), len(curve))
        if m > 0:
            plt.plot(ks_curve[:m], curve[:m], linewidth=1.8, label=name)
    plt.xlabel("k")
    plt.ylabel("Recall@k (macro average)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    return fig


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------- JSON dump & comparisons ----------

def save_run_json(
    path: str,
    ks_curve: np.ndarray,
    sections_payload: Dict[str, Dict],
    global_meta: Dict[str, int],
):
    """
    sections_payload: {
        "all": {
            "recall_curve": [...],
            "recall_table": [...],
            "metrics_table": [...],
            "num_queries": int,
            "num_eval_queries": int
        },
        "critical": { ... }
    }
    global_meta: e.g. {"max_k": 500}
    """
    payload = {
        "meta": {**global_meta},
        "ks_curve": [int(x) for x in ks_curve],
        "sections": sections_payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_comparison_runs(arg_list: Optional[List[str]]) -> List[Tuple[str, dict]]:
    """
    Parse name=path pairs and load JSON for each.
    Accepts up to 3 items (extra are ignored).
    Supports both new-format (with 'sections') and legacy-format (single section).
    """
    if not arg_list:
        return []
    runs = []
    for idx, item in enumerate(arg_list[:3]):
        if "=" not in item:
            raise ValueError(f"--compare-json expects NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        with open(path, "r", encoding="utf-8") as f:
            runs.append((name.strip(), json.load(f)))
    return runs


def _extract_section(js: dict, section_key: str) -> Optional[dict]:
    """
    Return section dict for given key ("all" or "critical").
    Backward-compat: if 'sections' missing, fabricate section from legacy keys.
    """
    if "sections" in js and isinstance(js["sections"], dict):
        return js["sections"].get(section_key)
    # Legacy fallback: treat single tables/curve as applicable to both
    sec = {}
    if "recall_curve" in js and isinstance(js["recall_curve"], list):
        sec["recall_curve"] = js["recall_curve"]
    if "recall_table" in js:
        sec["recall_table"] = js["recall_table"]
    if "metrics_table" in js:
        sec["metrics_table"] = js["metrics_table"]
    # optional meta
    meta = js.get("meta", {})
    sec["num_queries"] = meta.get("num_queries", 0)
    sec["num_eval_queries"] = meta.get("num_eval_queries", 0)
    return sec if sec else None


def merge_recall_tables(base_df: pd.DataFrame, comps: List[Tuple[str, dict]], section_key: str) -> pd.DataFrame:
    out = base_df.rename(columns={"Recall": "Recall (current)"})
    for name, js in comps:
        sec = _extract_section(js, section_key)
        if not sec:
            continue
        comp_df = pd.DataFrame(sec.get("recall_table", []))
        if "k" in comp_df and "Recall" in comp_df:
            comp_df = comp_df[["k", "Recall"]].copy()
            out = out.merge(comp_df, on="k", how="left")
            out.rename(columns={"Recall": f"Recall ({name})"}, inplace=True)
    return out


def merge_metric_tables(base_df: pd.DataFrame, comps: List[Tuple[str, dict]], section_key: str) -> pd.DataFrame:
    out = base_df.copy()
    out.rename(columns={"NDCG": "NDCG (current)", "MAP": "MAP (current)", "MRR": "MRR (current)"}, inplace=True)
    for name, js in comps:
        sec = _extract_section(js, section_key)
        if not sec:
            continue
        comp_df = pd.DataFrame(sec.get("metrics_table", []))
        if set(["k", "NDCG", "MAP", "MRR"]).issubset(comp_df.columns):
            comp_df = comp_df[["k", "NDCG", "MAP", "MRR"]].copy()
            comp_df.rename(columns={
                "NDCG": f"NDCG ({name})",
                "MAP": f"MAP ({name})",
                "MRR": f"MRR ({name})",
            }, inplace=True)
            out = out.merge(comp_df, on="k", how="left")
    return out


def style_headers_with_colors(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """
    Inject simple <span style="color:..."> tags into column headers for HTML.
    Terminal prints remain unaffected (we use unstyled frames there).
    """
    palette = ["#1f77b4", "#d62728", "#2ca02c"]  # blue, red, green
    cmap = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    new_cols = []
    for c in df.columns:
        if c.endswith("(current)"):
            base = c.rsplit("(", 1)[0].strip()
            new_cols.append(f'{base} (<span style="color:#000">current</span>)')
        else:
            if "(" in c and c.endswith(")"):
                base, paren = c.rsplit("(", 1)
                name = paren[:-1]
                if name in cmap:
                    new_cols.append(f'{base.strip()} (<span style="color:{cmap[name]}">{name}</span>)')
                else:
                    new_cols.append(c)
            else:
                new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df


# --------------------------
# HTML
# --------------------------

def print_ranking_pattern_analysis_terminal(
    queries: List[Query],
    per_query_all: List[PerQueryEval],
    per_query_crit: List[PerQueryEval],
    cutoffs: List[int] = [1, 3, 5, 10],
):
    """
    Print per-query ranking: first high-level pattern, then detailed list of documents.
    """
    qid_to_all = {pq.query_id: pq for pq in per_query_all}
    
    def make_pattern(binary_rels: List[int]) -> str:
        if not binary_rels:
            return "(no results)"
        return "".join("+" if rel == 1 else "-" for rel in binary_rels)
    
    print("\n" + "=" * 150)
    print("RANKING ANALYSIS")
    print("=" * 150)
    
    for q in queries:
        qid = q.query_id
        pq_all = qid_to_all.get(qid)
        
        if not pq_all:
            continue
        
        print(f"\nQuery: {q.text}")
        print(f"Ranking {len(pq_all.retrieved_doc_ids)} documents | {pq_all.num_relevant} relevant in ground truth")
        
        # High-level pattern
        pattern = make_pattern(pq_all.binary_rels)
        print(f"Pattern: {pattern}")
        
        # Summary recall
        found_at_cutoffs = []
        for k in cutoffs:
            found = sum(pq_all.binary_rels[:k])
            total = pq_all.num_relevant
            recall = (found / total * 100) if total > 0 else 0.0
            found_at_cutoffs.append(f"@{k}={found}/{total}({recall:.0f}%)")
        print(f"Recall:  {' '.join(found_at_cutoffs)}")
        
        print("-" * 150)
        
        # Detailed list of every document
        for rank, (doc_id, is_relevant, category, snippet, url) in enumerate(zip(
            pq_all.retrieved_doc_ids,
            pq_all.binary_rels,
            pq_all.categories,
            pq_all.snippets,
            pq_all.urls
        ), start=1):
            relevant_marker = "[✓]" if is_relevant else "[ ]"
            snippet_display = snippet if snippet else "(no text)"
            url_display = url if url else "(no url)"
            # Category can be quite long now with multiple categories joined by " + "
            print(f" {rank:3d}. {relevant_marker} Category={category:<50}")
            print(f"         URL: {url_display}")
            print(f"         Text: {repr(snippet_display)}")
        
        print("=" * 150)


def make_html_report_two_sections(
    plots: Dict[str, str],  # {"all": base64_png, "critical": base64_png}
    tables: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],  # {"all": (recall_df, metrics_df), ...}
    counts: Dict[str, Tuple[int, int]],  # {"all": (num_queries, num_eval_queries), ...}
) -> str:
    def df_to_html_table(df: pd.DataFrame) -> str:
        # allow colored header spans
        return df.to_html(index=False, escape=False, float_format=lambda x: f"{x:.4f}")

    section_titles = {
        "all": "Section A — Using positive_corpus_ids_all",
        "critical": "Section B — Using positive_corpus_ids_critical_only",
    }

    blocks = []
    for key in ["all", "critical"]:
        if key not in plots or key not in tables or key not in counts:
            # skip if missing
            continue
        base64_png = plots[key]
        recall_df, metrics_df = tables[key]
        num_queries, num_eval_queries = counts[key]

        block = f"""
        <div class="card">
          <h2>{section_titles[key]}</h2>
          <div class="meta">
            <div>Total queries in dataset: <b>{num_queries}</b></div>
            <div>Queries with ≥1 relevant doc: <b>{num_eval_queries}</b></div>
          </div>

          <div class="card">
            <h3>Recall@k Curve (k=0..500)</h3>
            <img src="data:image/png;base64,{base64_png}" alt="Recall@k curve - {key}"/>
            <div class="muted">Macro-averaged recall across queries.</div>
          </div>

          <div class="card">
            <h3>Recall Table</h3>
            {df_to_html_table(recall_df)}
          </div>

          <div class="card">
            <h3>NDCG / MAP / MRR</h3>
            {df_to_html_table(metrics_df)}
          </div>
        </div>
        """
        blocks.append(block)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WHO API IR Evaluation Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
 h1, h2, h3 {{ margin-bottom: 8px; }}
 .meta {{ color: #555; margin-bottom: 16px; }}
 .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
 img {{ max-width: 100%; height: auto; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
 th {{ background: #f6f6f6; }}
 td:first-child, th:first-child {{ text-align: left; }}
 .muted {{ color: #777; font-size: 0.95em; }}
</style>
</head>
<body>
  <h1>WHO API IR Evaluation</h1>
  {"".join(blocks)}
  <div class="muted">
    <p>Binary relevance from qrels; NDCG uses log2 discount; MAP@k normalizes by min(|relevant|, k); MRR is reciprocal of first relevant rank within cutoff.</p>
  </div>
</body>
</html>
"""
    return html


# --------------------------
# Main runner
# --------------------------

def build_qid_to_results(
    queries: List[Query],
    base_url: str,
    get_doc_id_fn: Callable[[str], int],
    throttle: float = 0.0,
    timeout: float = 30.0,
) -> Tuple[Dict[int, List[int]], Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
    qid_to_results: Dict[int, List[int]] = {}
    qid_to_categories: Dict[int, List[str]] = {}
    qid_to_snippets: Dict[int, List[str]] = {}
    qid_to_urls: Dict[int, List[str]] = {}

    for q in tqdm(queries, desc="Querying API"):
        payload = call_api(base_url, q.text, timeout=timeout)

        # Expect a JSON dict with key 'content' -> list of result dicts
        results = payload.get("content", [])
        ranked_doc_ids: List[int] = []
        ranked_categories: List[str] = []
        ranked_snippets: List[str] = []
        ranked_urls: List[str] = []
        
        for item in results:
            # Prefer 'schema_object' string if present; else pass the whole item as JSON string.
            if isinstance(item, dict) and "schema_object" in item and isinstance(item["schema_object"], str):
                schema_obj_str = item["schema_object"]
            else:
                # Pass the entire result as JSON for user-defined parsing
                schema_obj_str = json.dumps(item, ensure_ascii=False)

            try:
                docid = int(get_doc_id_fn(schema_obj_str))
                ranked_doc_ids.append(docid)
                
                # Extract category, snippet, and url from schema_object
                try:
                    schema_obj = json.loads(schema_obj_str) if isinstance(schema_obj_str, str) else schema_obj_str
                    
                    # WHO categories is always a list of strings
                    categories = schema_obj.get("who_categories", ["unknown"])
                    category_str = " + ".join(categories)
                    ranked_categories.append(category_str)
                    
                    # Extract text snippet - try multiple fields
                    snippet = (
                        schema_obj.get("description") or
                        schema_obj.get("name") or
                        schema_obj.get("text") or
                        ""
                    )
                    if isinstance(snippet, list) and len(snippet) > 0:
                        snippet = snippet[0]
                    # Truncate to 100 chars
                    snippet_str = str(snippet)[:100]
                    ranked_snippets.append(snippet_str)
                    
                    # Extract URL
                    url = schema_obj.get("url", "")
                    ranked_urls.append(str(url))
                except Exception:
                    ranked_categories.append("unknown")
                    ranked_snippets.append("")
                    ranked_urls.append("")
            except Exception:
                # Skip items that fail mapping
                continue

        qid_to_results[q.query_id] = ranked_doc_ids
        qid_to_categories[q.query_id] = ranked_categories
        qid_to_snippets[q.query_id] = ranked_snippets
        qid_to_urls[q.query_id] = ranked_urls

        if throttle > 0:
            time.sleep(throttle)

    return qid_to_results, qid_to_categories, qid_to_snippets, qid_to_urls


def main():
    parser = argparse.ArgumentParser(description="Evaluate local WHO API against HF qrels.")
    parser.add_argument("--dataset", type=str, default="withpi/nlweb_who_achint_qrel",
                        help="HF dataset name housing qrels/queries/corpus.")
    parser.add_argument("--queries-config", type=str, default="queries", help="Queries config name.")
    parser.add_argument("--qrel-config", type=str, default="achint-qrels", help="Qrel config name.")
    parser.add_argument("--split", type=str, default="train", help="Split for configs.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/who",
                        help="Local WHO endpoint base URL.")
    parser.add_argument("--docid-func", type=str, default=None,
                        help="Optional 'module:function' for get_doc_id(schema_object: str)->int. "
                             "If omitted, uses built-in default URL mapper.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries per call.")
    parser.add_argument("--backoff", type=float, default=1.0, help="HTTP retry backoff base.")
    parser.add_argument("--max-k", type=int, default=100, help="Max k for recall curve.")
    parser.add_argument("--output", type=str, default="who_eval_report.html", help="Output HTML report path.")
    parser.add_argument("--throttle", type=float, default=0.0, help="Seconds to sleep between queries.")
    parser.add_argument("--interactive-plot", action="store_true",
                        help="Also show a non-blocking interactive plot window (Section A only).")
    parser.add_argument("--dump-json", type=str, default=None,
                        help="If set, dump all metrics/tables/curves to this JSON file.")
    parser.add_argument("--compare-json", type=str, nargs="*", default=None,
                        help="Optional NAME=PATH pairs of prior runs to compare (up to 3).")

    args = parser.parse_args()

    # Load queries + results
    print("Loading queries...")
    queries = load_queries(
        dataset_name=args.dataset,
        queries_config=args.queries_config,
        split=args.split,
    )
    print(f"Loaded {len(queries)} queries.")

    # Prepare docid mapper
    if args.docid_func:
        get_doc_id_fn = load_docid_func(args.docid_func)
    else:
        get_doc_id_fn = default_get_doc_id

    # Run evaluation queries
    qid_to_results, qid_to_categories, qid_to_snippets, qid_to_urls = build_qid_to_results(
        queries=queries,
        base_url=args.base_url,
        get_doc_id_fn=get_doc_id_fn,
        throttle=args.throttle,
        timeout=args.timeout,
    )

    # Load qrels for both sections
    print("Loading qrels (all / critical)...")
    qrels_all = load_qrels_by_field(args.dataset, args.qrel_config, args.split, "positive_corpus_ids_all")
    qrels_critical = load_qrels_by_field(args.dataset, args.qrel_config, args.split, "positive_corpus_ids_critical_only")

    # Evaluate for ALL
    print("Evaluating (ALL)...")
    ks_curve_all, recall_df_all, metrics_df_all, per_query_all, dense_all = evaluate(
        qid_to_gt=qrels_all,
        qid_to_results=qid_to_results,
        qid_to_categories=qid_to_categories,
        qid_to_snippets=qid_to_snippets,
        qid_to_urls=qid_to_urls,
        max_k_for_curve=args.max_k,
    )
    num_eval_queries_all = len([pq for pq in per_query_all if pq.num_relevant > 0])

    # Evaluate for CRITICAL
    print("Evaluating (CRITICAL)...")
    ks_curve_crit, recall_df_crit, metrics_df_crit, per_query_crit, dense_crit = evaluate(
        qid_to_gt=qrels_critical,
        qid_to_results=qid_to_results,
        qid_to_categories=qid_to_categories,
        qid_to_snippets=qid_to_snippets,
        qid_to_urls=qid_to_urls,
        max_k_for_curve=args.max_k,
    )
    num_eval_queries_crit = len([pq for pq in per_query_crit if pq.num_relevant > 0])

    # Sanity: ks_curve should be identical; if not, align to min length
    ks_curve = ks_curve_all if len(ks_curve_all) <= len(ks_curve_crit) else ks_curve_crit
    if len(dense_all) != len(ks_curve):
        dense_all = dense_all[:len(ks_curve)]
    if len(dense_crit) != len(ks_curve):
        dense_crit = dense_crit[:len(ks_curve)]

    # Load comparison runs (if any)
    comp_runs = load_comparison_runs(args.compare_json)

    # Build comparison curves per section
    comp_curves_all: List[Tuple[str, np.ndarray]] = []
    comp_curves_crit: List[Tuple[str, np.ndarray]] = []
    for name, js in comp_runs:
        # ks alignment (optional)
        # We assume same 0..K; if not, just truncate to min length below.
        sec_all = _extract_section(js, "all")
        sec_crit = _extract_section(js, "critical")
        rc_all = np.array(sec_all["recall_curve"], dtype=float) if sec_all and "recall_curve" in sec_all else None
        rc_crit = np.array(sec_crit["recall_curve"], dtype=float) if sec_crit and "recall_curve" in sec_crit else None

        if rc_all is None and rc_crit is None:
            # legacy: single curve at top level?
            rc_legacy = np.array(js.get("recall_curve", []), dtype=float)
            if len(rc_legacy) > 0:
                rc_all = rc_legacy.copy()
                rc_crit = rc_legacy.copy()

        if rc_all is not None and len(rc_all) > 0:
            comp_curves_all.append((name, rc_all[:len(ks_curve)]))
        if rc_crit is not None and len(rc_crit) > 0:
            comp_curves_crit.append((name, rc_crit[:len(ks_curve)]))

    # Plot recall curves (both sections)
    print("Rendering recall@k plots...")
    fig_all = plot_recall_curve_multi(
        ks_curve=ks_curve,
        current_dense=dense_all,
        comp_curves=comp_curves_all,
        title="Recall@k (macro average) — ALL",
    )
    fig_crit = plot_recall_curve_multi(
        ks_curve=ks_curve,
        current_dense=dense_crit,
        comp_curves=comp_curves_crit,
        title="Recall@k (macro average) — CRITICAL",
    )

    # Save figures & embed
    png_b64_all = fig_to_base64_png(fig_all)
    png_b64_crit = fig_to_base64_png(fig_crit)
    fig_path_all = args.output.replace(".html", "_recall_all.png")
    fig_path_crit = args.output.replace(".html", "_recall_critical.png")
    fig_all.savefig(fig_path_all, dpi=150, bbox_inches="tight")
    fig_crit.savefig(fig_path_crit, dpi=150, bbox_inches="tight")

    # Optional interactive (Section A / ALL only)
    if args.interactive_plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")  # may fail headless
            plt.figure(figsize=(8,5))
            plt.plot(ks_curve, dense_all, linewidth=2, label="current (ALL)")
            for name, curve in comp_curves_all:
                m = min(len(ks_curve), len(curve))
                if m > 0:
                    plt.plot(ks_curve[:m], curve[:m], linewidth=1.8, label=name)
            plt.xlabel("k")
            plt.ylabel("Recall@k (macro avg)")
            plt.title("Recall@k (macro average) — ALL")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.legend()
            plt.ion()
            plt.show(block=False)
        except Exception:
            pass

    # Print ranking pattern analysis to terminal
    print_ranking_pattern_analysis_terminal(
        queries=queries,
        per_query_all=per_query_all,
        per_query_crit=per_query_crit,
        cutoffs=[1, 10, 100],
    )

    # Print tables to terminal (current only for brevity)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Recall Table — ALL ===")
    print(recall_df_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== NDCG / MAP / MRR — ALL ===")
    print(metrics_df_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Recall Table — CRITICAL ===")
    print(recall_df_crit.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== NDCG / MAP / MRR — CRITICAL ===")
    print(metrics_df_crit.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Also print comparisons to terminal if provided
    if comp_runs:
        print("\n=== Recall Table (with comparisons) — ALL ===")
        print(merge_recall_tables(recall_df_all, comp_runs, "all").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n=== NDCG / MAP / MRR (with comparisons) — ALL ===")
        print(merge_metric_tables(metrics_df_all, comp_runs, "all").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n=== Recall Table (with comparisons) — CRITICAL ===")
        print(merge_recall_tables(recall_df_crit, comp_runs, "critical").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n=== NDCG / MAP / MRR (with comparisons) — CRITICAL ===")
        print(merge_metric_tables(metrics_df_crit, comp_runs, "critical").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Dump JSON if requested
    if args.dump_json:
        sections_payload = {
            "all": {
                "recall_curve": [float(x) for x in dense_all],
                "recall_table": recall_df_all.to_dict(orient="records"),
                "metrics_table": metrics_df_all.to_dict(orient="records"),
                "num_queries": len(queries),
                "num_eval_queries": num_eval_queries_all,
            },
            "critical": {
                "recall_curve": [float(x) for x in dense_crit],
                "recall_table": recall_df_crit.to_dict(orient="records"),
                "metrics_table": metrics_df_crit.to_dict(orient="records"),
                "num_queries": len(queries),
                "num_eval_queries": num_eval_queries_crit,
            },
        }
        global_meta = {
            "max_k": int(ks_curve.max() if len(ks_curve) else 0),
        }
        save_run_json(
            path=args.dump_json,
            ks_curve=ks_curve,
            sections_payload=sections_payload,
            global_meta=global_meta,
        )
        print(f"Saved run JSON to: {args.dump_json}")

    # Build comparison-augmented tables for HTML (both sections)
    comp_names = [name for name, _ in comp_runs]
    recall_all_html = merge_recall_tables(recall_df_all, comp_runs, "all")
    metrics_all_html = merge_metric_tables(metrics_df_all, comp_runs, "all")
    recall_crit_html = merge_recall_tables(recall_df_crit, comp_runs, "critical")
    metrics_crit_html = merge_metric_tables(metrics_df_crit, comp_runs, "critical")

    recall_all_html = style_headers_with_colors(recall_all_html, comp_names)
    metrics_all_html = style_headers_with_colors(metrics_all_html, comp_names)
    recall_crit_html = style_headers_with_colors(recall_crit_html, comp_names)
    metrics_crit_html = style_headers_with_colors(metrics_crit_html, comp_names)

    # Write HTML report (two sections)
    html = make_html_report_two_sections(
        plots={
            "all": png_b64_all,
            "critical": png_b64_crit,
        },
        tables={
            "all": (recall_all_html, metrics_all_html),
            "critical": (recall_crit_html, metrics_crit_html),
        },
        counts={
            "all": (len(queries), num_eval_queries_all),
            "critical": (len(queries), num_eval_queries_crit),
        },
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport written to: {args.output}")
    print(f"Recall plot images: {fig_path_all}, {fig_path_crit}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)