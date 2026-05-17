"""Build the ViDoRe-TR community split (first public Turkish visual document retrieval benchmark).

Placeholder — Day 9-10, depends on OPEN_ITEMS.md A3 (BEIR format + annotation methodology).

Schema (MTEB BEIR-compatible):
- corpus.jsonl   — {"_id": str, "title": str, "text": str, "image_path": str}
- queries.jsonl  — {"_id": str, "text": str}  (Turkish queries)
- qrels/test.tsv — query_id\tcorpus_id\trelevance(0/1/2)

Target size (v0.1 preview):
- 200-300 page-query pairs minimum
- Sources: mevzuat + KAP + YÖK + synthetic invoice mix
- Manual annotation: Mert + 2 optional collaborators; IAA check on subset
- Query generation: Qwen3-32B prompted + 100% human spot-check (no hallucinated queries)

Output: data/vidore-tr/ + HF dataset push Verm1ion/ViDoRe-TR-eval (private → public after PR-ready)
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Day 9-10 task. See OPEN_ITEMS.md A3.")


if __name__ == "__main__":
    main()
