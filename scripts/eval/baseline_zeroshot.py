"""Day-1 critical: ColQwen2.5-3B-multilingual zero-shot baseline on ViDoRe V3.

Placeholder — Day 1 implementation (CRITICAL — first concrete output).
See OPEN_ITEMS.md A9.

Pipeline:
1. Load Metric-AI/ColQwen2.5-3b-multilingual-v1.0 via colpali-engine
2. Load ViDoRe V3 8 public subtasks via vidore-benchmark
3. Run zero-shot eval: NDCG@10, recall@1/5/10, mAP
4. Log to W&B (run group: 'baseline')
5. Save results to eval/results/baseline_zeroshot_vidore_v3.json
6. Append decision log entry: "ColQwen2.5-3B baseline NDCG@10 = X.XX"
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Day 1 CRITICAL task. See OPEN_ITEMS.md A9.")


if __name__ == "__main__":
    main()
