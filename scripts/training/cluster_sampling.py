"""Cluster-based positive sampling — Nemotron §3.3.3 verbatim verified.

Nemotron ColEmbed V2 §3.3.3 quote (2026-05-18 cross-validation):
    "followed by a K-Means clustering approach utilizing gap statistics
    [tibshirani2001estimating] to choose the most representative k clusters.
    To ensure diversity and balance the domains of our training blend,
    we perform uniform sampling the 14 discovered clusters."

ColTurk-VDR adaptation: page embeddings → PCA-50 → K-Means with gap statistics
(default k=14 matching paper; let gap_stat choose 8-20 range) → batch-level
weighted random sampler.

Beklenen lift: +0.3 ila +0.7 NDCG@10 (Nemotron implied, no explicit ablation
number in paper). ColTurk-VDR dataset Nemotron'dan dar; üst sınır marjinal.

KARARLAR S31: cluster sampling P1, k=14 paper default, weighted by sqrt(size).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("cluster_sampling")

DEFAULT_K = 14
DEFAULT_PCA_DIM = 50
GAP_STAT_K_RANGE = range(8, 21)


def compute_gap_statistic(embeddings: np.ndarray, k_range: range, n_refs: int = 10, seed: int = 42) -> int:
    """Tibshirani 2001 gap statistic to choose k.

    For each candidate k:
        gap(k) = log(W_k_ref_mean) - log(W_k_observed)
    where W_k is within-cluster sum of squares. Pick first k where
    gap(k) >= gap(k+1) - s_{k+1} (one-standard-error rule).
    """
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]

    rng = np.random.default_rng(seed)
    log_w_observed: list[float] = []
    log_w_refs: list[list[float]] = []

    minima = embeddings.min(axis=0)
    maxima = embeddings.max(axis=0)
    n_samples, n_dims = embeddings.shape

    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        labels = km.fit_predict(embeddings)
        w = sum(((embeddings[labels == c] - km.cluster_centers_[c]) ** 2).sum() for c in range(k))
        log_w_observed.append(np.log(w + 1e-12))

        ref_logs: list[float] = []
        for _ in range(n_refs):
            ref = rng.uniform(minima, maxima, size=(n_samples, n_dims))
            km_ref = KMeans(n_clusters=k, n_init="auto", random_state=seed)
            ref_labels = km_ref.fit_predict(ref)
            w_ref = sum(
                ((ref[ref_labels == c] - km_ref.cluster_centers_[c]) ** 2).sum() for c in range(k)
            )
            ref_logs.append(np.log(w_ref + 1e-12))
        log_w_refs.append(ref_logs)

    gaps = np.array([np.mean(refs) - obs for refs, obs in zip(log_w_refs, log_w_observed)])
    s = np.array([np.std(refs) * np.sqrt(1 + 1.0 / n_refs) for refs in log_w_refs])
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - s[i + 1]:
            return list(k_range)[i]
    return list(k_range)[int(np.argmax(gaps))]


def fit_clusters(
    embeddings: np.ndarray,
    k: int | None = DEFAULT_K,
    pca_dim: int = DEFAULT_PCA_DIM,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce to PCA-50, run K-Means, return (labels, centroids, k_used)."""
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]
    from sklearn.decomposition import PCA  # type: ignore[import-not-found]

    n_samples, n_dims = embeddings.shape
    effective_pca = min(pca_dim, n_dims, n_samples - 1)
    LOG.info("PCA: %d-dim -> %d-dim", n_dims, effective_pca)
    pca = PCA(n_components=effective_pca, random_state=seed)
    reduced = pca.fit_transform(embeddings)

    if k is None:
        LOG.info("Choosing k via gap statistic (range %s)...", GAP_STAT_K_RANGE)
        k = compute_gap_statistic(reduced, GAP_STAT_K_RANGE, seed=seed)
        LOG.info("Gap statistic chose k=%d", k)
    else:
        LOG.info("Using paper-default k=%d", k)

    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(reduced)
    return labels, km.cluster_centers_, k


def cluster_weights(labels: np.ndarray, scheme: str = "sqrt") -> np.ndarray:
    """Per-sample sampling weight from cluster assignment.

    scheme:
        - "uniform": every cluster sampled equally (Nemotron default)
        - "sqrt": cluster_size^-0.5 (balanced; avoids tiny clusters drowning)
        - "inverse": 1/cluster_size (aggressive rebalancing)
    """
    unique, counts = np.unique(labels, return_counts=True)
    if scheme == "uniform":
        per_cluster_w = {c: 1.0 / len(unique) for c in unique}
    elif scheme == "sqrt":
        inv_sqrt = {c: 1.0 / np.sqrt(count) for c, count in zip(unique, counts)}
        z = sum(inv_sqrt.values())
        per_cluster_w = {c: w / z for c, w in inv_sqrt.items()}
    elif scheme == "inverse":
        inv = {c: 1.0 / count for c, count in zip(unique, counts)}
        z = sum(inv.values())
        per_cluster_w = {c: w / z for c, w in inv.items()}
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")

    return np.array([per_cluster_w[c] for c in labels], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="K-means k=14 cluster sampling weights for Stage-2 trainer")
    parser.add_argument("--embeddings", type=Path, required=True, help=".npz: page_id -> embedding")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON: page_id -> {cluster, weight}")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of clusters (default: {DEFAULT_K})")
    parser.add_argument("--auto-k", action="store_true", help="Choose k via gap statistic (overrides --k)")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM)
    parser.add_argument("--scheme", choices=["uniform", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz = np.load(args.embeddings)
    page_ids = list(npz.files)
    embeddings = np.stack([npz[k].mean(axis=0) if npz[k].ndim > 1 else npz[k] for k in page_ids]).astype(
        np.float32
    )

    k = None if args.auto_k else args.k
    labels, centroids, k_used = fit_clusters(embeddings, k=k, pca_dim=args.pca_dim, seed=args.seed)
    weights = cluster_weights(labels, scheme=args.scheme)

    out = {
        pid: {"cluster": int(label), "weight": float(weight)}
        for pid, label, weight in zip(page_ids, labels, weights)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"k": int(k_used), "scheme": args.scheme, "assignments": out}, indent=2))
    LOG.info("Wrote cluster assignments for %d pages, k=%d, scheme=%s", len(page_ids), k_used, args.scheme)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
