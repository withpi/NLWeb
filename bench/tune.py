#!/usr/bin/env python3
"""
Hyperparameter tuning script for WHO handler.

Optimizes reranking weights and thresholds by running benchmarks with different configurations
and tracking Recall@5 metric.

Usage:
    python bench/tune.py --output tune_results.json
"""

import argparse
import json
import sys
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Add bench directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_benchmark_v2 import (
    Query,
    load_queries,
    load_qrels_by_field,
    build_qid_to_results,
    evaluate,
    default_get_doc_id,
)

# Dev set query IDs (duplicated from run_benchmark_v2.py)
DEV_QUERY_IDS = [
    "3",   # Product: gifts for toddlers (broad product category)
    "12",  # Product: Mediterranean herbs from Greece (specific food product with geographic constraint)
    "14",  # Non-product: movies for teenagers (entertainment/media)
    "15",  # Recipe: jam filled gluten free cake recipe (simple recipe query)
    "16",  # Non-product: trails with olive trees in italy (travel/location-based)
    "20",  # Product: Porcelain clay for tea cups (specialized equipment/supplies)
    "24",  # Recipe: japanese dishes with rice and fish (cuisine + constraints)
]


@dataclass
class HyperparamConfig:
    ce_weight: float  # Weight for cross-encoder score in final ranking
    min_score: int  # Minimum score threshold to include results
    query_classification_threshold: float  # Threshold for query classification
    k: int  # Number of results to fetch from corpus
    config_name: str  # Human-readable name for this config
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ce_weight": self.ce_weight,
            "min_score": self.min_score,
            "query_classification_threshold": self.query_classification_threshold,
            "k": self.k,
            "config_name": self.config_name,
        }


@dataclass
class TuneResult:
    config: HyperparamConfig
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    map_at_10: float
    perfect_recall_rate: float  # Fraction of queries achieving 100% recall@k
    num_eval_queries: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "recall_at_20": self.recall_at_20,
            "ndcg_at_10": self.ndcg_at_10,
            "map_at_10": self.map_at_10,
            "perfect_recall_rate": self.perfect_recall_rate,
            "num_eval_queries": self.num_eval_queries,
        }


def extract_recall_at_k(recall_df: pd.DataFrame, k: int) -> float:
    """Extract Recall@k from the recall dataframe."""
    row = recall_df[recall_df["k"] == k]
    if len(row) == 0:
        return 0.0
    return float(row["Recall"].iloc[0])


def extract_metric_at_k(metrics_df: pd.DataFrame, k: int, metric: str) -> float:
    """Extract a specific metric@k from the metrics dataframe."""
    row = metrics_df[metrics_df["k"] == k]
    if len(row) == 0:
        return 0.0
    return float(row[metric].iloc[0])


def compute_perfect_recall_rate(per_query, k: int) -> float:
    """
    Fraction of queries that achieve 100% recall at k.
    For each query, we check if all relevant documents are in top-k results.
    """
    from run_benchmark_v2 import recall_at_k
    
    perfect_count = 0
    for pq in per_query:
        if pq.num_relevant == 0:
            continue
        recall = recall_at_k(pq.binary_rels, k, pq.num_relevant)
        if recall >= 1.0:
            perfect_count += 1
    
    total = len([pq for pq in per_query if pq.num_relevant > 0])
    if total == 0:
        return 0.0
    return float(perfect_count) / float(total)


def run_benchmark_with_config(
    config: HyperparamConfig,
    queries: list[Query],
    qrels_all: dict[int, set],
    base_url: str,
    timeout: float,
    run_number: int,
    total_runs: int,
) -> TuneResult:
    """Run benchmark with a specific hyperparameter configuration."""
    print(f"\n[{run_number}/{total_runs}] Testing: {config.config_name}")
    print(f"  k={config.k}, ce_weight={config.ce_weight}, min_score={config.min_score}, qc_thresh={config.query_classification_threshold}")
    
    # Build URL with hyperparameters as query params
    # For baseline, don't add params - use server defaults
    # For baseline_explicit, pass params explicitly to verify param passing works
    if config.config_name == "baseline":
        url_with_params = base_url
        print(f"  Using URL: {url_with_params} (no params - using server defaults)")
    else:
        separator = "&" if "?" in base_url else "?"
        url_with_params = (
            f"{base_url}{separator}"
            f"ce_weight={config.ce_weight}&"
            f"min_score={config.min_score}&"
            f"query_classification_threshold={config.query_classification_threshold}&"
            f"num={config.k}"
        )
        if config.config_name == "baseline_explicit":
            print(f"  Using URL: {url_with_params} (explicit params - should match baseline)")
        else:
            print(f"  Using URL: {url_with_params}")
    
    # Run benchmark
    qid_to_results, qid_to_categories, qid_to_snippets, qid_to_urls, _ = build_qid_to_results(
        queries=queries,
        base_url=url_with_params,
        get_doc_id_fn=default_get_doc_id,
        throttle=0.0,
        timeout=timeout,
    )
    
    # Evaluate
    _, recall_df, metrics_df, per_query, _ = evaluate(
        qid_to_gt=qrels_all,
        qid_to_results=qid_to_results,
        qid_to_categories=qid_to_categories,
        qid_to_snippets=qid_to_snippets,
        qid_to_urls=qid_to_urls,
        max_k_for_curve=100,
    )
    
    # Extract metrics
    recall_at_5 = extract_recall_at_k(recall_df, 5)
    recall_at_10 = extract_recall_at_k(recall_df, 10)
    recall_at_20 = extract_recall_at_k(recall_df, 20)
    ndcg_at_10 = extract_metric_at_k(metrics_df, 10, "NDCG")
    map_at_10 = extract_metric_at_k(metrics_df, 10, "MAP")
    perfect_recall_rate = compute_perfect_recall_rate(per_query, 100)
    num_eval_queries = len([pq for pq in per_query if pq.num_relevant > 0])
    
    result = TuneResult(
        config=config,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_20,
        ndcg_at_10=ndcg_at_10,
        map_at_10=map_at_10,
        perfect_recall_rate=perfect_recall_rate,
        num_eval_queries=num_eval_queries,
    )
    
    print(f"  R@5={recall_at_5:.3f} R@10={recall_at_10:.3f} R@20={recall_at_20:.3f} | NDCG@10={ndcg_at_10:.3f} | PerfectRecall={perfect_recall_rate:.3f}")
    
    return result


def generate_hyperparam_configs() -> list[HyperparamConfig]:
    """
    Generate hyperparameter configurations to test.
    
    Starts with faster configs (smaller k) and explores reranking weights.
    """
    configs = []
    
    # Baseline (current defaults - no params passed)
    configs.append(HyperparamConfig(
        ce_weight=0.4,
        min_score=1,
        query_classification_threshold=0.4,
        k=60,
        config_name="baseline",
    ))
    
    # Baseline_explicit (pass default params explicitly to verify param passing works)
    configs.append(HyperparamConfig(
        ce_weight=0.4,
        min_score=1,
        query_classification_threshold=0.4,
        k=60,
        config_name="baseline_explicit",
    ))
    
    # Vary k (number of results to fetch) - START WITH FASTEST
    for k in [20, 30, 40, 60, 80, 100]:
        configs.append(HyperparamConfig(
            ce_weight=0.4,
            min_score=1,
            query_classification_threshold=0.4,
            k=k,
            config_name=f"k_{k}",
        ))
    
    # Vary ce_weight (weight of cross-encoder in final score)
    for ce_weight in [0.0, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0]:
        configs.append(HyperparamConfig(
            ce_weight=ce_weight,
            min_score=1,
            query_classification_threshold=0.4,
            k=60,
            config_name=f"ce_weight_{ce_weight:.1f}",
        ))
    
    # Vary min_score threshold (test different thresholds from very low to moderate)
    for min_score in [1, 10, 20, 30, 40, 50]:
        configs.append(HyperparamConfig(
            ce_weight=0.4,
            min_score=min_score,
            query_classification_threshold=0.4,
            k=60,
            config_name=f"min_score_{min_score}",
        ))
    
    # Vary query classification threshold
    for qc_thresh in [0.2, 0.3, 0.5, 0.6]:
        configs.append(HyperparamConfig(
            ce_weight=0.4,
            min_score=1,
            query_classification_threshold=qc_thresh,
            k=60,
            config_name=f"qc_thresh_{qc_thresh:.1f}",
        ))
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Tune WHO handler hyperparameters")
    parser.add_argument("--dataset", type=str, default="withpi/nlweb_who_achint_qrel",
                        help="HF dataset name")
    parser.add_argument("--queries-config", type=str, default="queries",
                        help="Queries config name")
    parser.add_argument("--qrel-config", type=str, default="achint-qrels",
                        help="Qrel config name")
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="HF dataset split name")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/who",
                        help="Local WHO endpoint base URL")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout seconds")
    parser.add_argument("--output", type=str, default="tune_results.json",
                        help="Output JSON path for results")
    parser.add_argument("--split", type=str, default="train",
                        help="Query split to use: 'train' (excludes dev), 'dev', 'test', 'dev,test', or 'all'")
    parser.add_argument("--optimize-metric", type=str, default="recall_at_5",
                        choices=["recall_at_5", "recall_at_10", "recall_at_20", "ndcg_at_10", "map_at_10", "perfect_recall_rate"],
                        help="Metric to optimize")
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config only (e.g., 'baseline', 'k_20', 'ce_weight_0.5')")
    
    args = parser.parse_args()
    
    print("="*80)
    print("WHO HANDLER HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load queries
    print("\nLoading queries...")
    all_queries = load_queries(
        dataset_name=args.dataset,
        queries_config=args.queries_config,
        split=args.dataset_split,
    )
    print(f"Loaded {len(all_queries)} total queries from dataset")
    
    # Filter queries based on --split parameter
    dev_qids = set(int(qid) for qid in DEV_QUERY_IDS)
    
    if args.split == "train":
        queries = [q for q in all_queries if q.query_id not in dev_qids]
        split_desc = "TRAIN (excludes dev)"
    elif args.split == "dev":
        queries = [q for q in all_queries if q.query_id in dev_qids]
        split_desc = "DEV"
    elif args.split in ["dev,test", "test,dev"]:
        queries = [q for q in all_queries if q.query_id in dev_qids]
        split_desc = "DEV (test is same as dev for now)"
    elif args.split == "test":
        queries = [q for q in all_queries if q.query_id in dev_qids]
        split_desc = "TEST (same as dev for now)"
    elif args.split == "all":
        queries = all_queries
        split_desc = "ALL"
    else:
        print(f"Unknown split: {args.split}, using train")
        queries = [q for q in all_queries if q.query_id not in dev_qids]
        split_desc = "TRAIN"
    
    print(f"Using {split_desc}: {len(queries)} queries")
    
    # Load qrels
    print("Loading qrels...")
    qrels_all = load_qrels_by_field(
        args.dataset,
        args.qrel_config,
        args.dataset_split,
        "positive_corpus_ids_all"
    )
    
    # Generate configurations to test
    all_configs = generate_hyperparam_configs()
    
    # Filter to specific config if requested
    if args.config:
        configs = [c for c in all_configs if c.config_name == args.config]
        if not configs:
            print(f"ERROR: Config '{args.config}' not found")
            print(f"Available configs: {', '.join(c.config_name for c in all_configs)}")
            sys.exit(1)
        print(f"\nRunning specific config: {args.config}")
    else:
        configs = all_configs
        print(f"\nWill test {len(configs)} configurations")
    
    print(f"Optimizing for: {args.optimize_metric}")
    print(f"Split: {split_desc}")
    
    # Run benchmarks for each config
    results: list[TuneResult] = []
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        try:
            result = run_benchmark_with_config(
                config=config,
                queries=queries,
                qrels_all=qrels_all,
                base_url=args.base_url,
                timeout=args.timeout,
                run_number=i,
                total_runs=len(configs),
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"Tuning completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("="*80)
    
    # Sort by optimization metric
    def get_metric_value(r: TuneResult) -> float:
        return getattr(r, args.optimize_metric)
    
    results_sorted = sorted(results, key=get_metric_value, reverse=True)
    
    # Print top 10 results
    print("\n" + "="*80)
    print(f"TOP 10 CONFIGURATIONS (by {args.optimize_metric})")
    print("="*80)
    print(f"{'Rank':<6} {'Config':<25} {args.optimize_metric:<14} {'R@5':<8} {'R@10':<8} {'PerfectR':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(results_sorted[:10], 1):
        opt_val = get_metric_value(result)
        print(f"{rank:<6} {result.config.config_name:<25} {opt_val:<14.4f} "
              f"{result.recall_at_5:<8.4f} {result.recall_at_10:<8.4f} {result.perfect_recall_rate:<10.4f}")
    
    # Best config
    best_result = results_sorted[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Config: {best_result.config.config_name}")
    print(f"  k: {best_result.config.k}")
    print(f"  ce_weight: {best_result.config.ce_weight}")
    print(f"  min_score: {best_result.config.min_score}")
    print(f"  query_classification_threshold: {best_result.config.query_classification_threshold}")
    print("\nMetrics:")
    print(f"  Recall@5:  {best_result.recall_at_5:.4f}")
    print(f"  Recall@10: {best_result.recall_at_10:.4f}")
    print(f"  Recall@20: {best_result.recall_at_20:.4f}")
    print(f"  NDCG@10:   {best_result.ndcg_at_10:.4f}")
    print(f"  MAP@10:    {best_result.map_at_10:.4f}")
    print(f"  Perfect Recall Rate: {best_result.perfect_recall_rate:.4f}")
    
    # Save results to JSON
    output_data = {
        "optimization_metric": args.optimize_metric,
        "query_split": args.split,
        "split_description": split_desc,
        "num_queries": len(queries),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "best_config": best_result.to_dict(),
        "all_results": [r.to_dict() for r in results_sorted],
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

