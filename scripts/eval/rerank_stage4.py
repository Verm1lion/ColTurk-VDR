"""Stage-4 cross-encoder reranker — top-1 attack'ın en büyük tek-leverage'ı.

Nemotron ColEmbed V2 paper §5.2 verbatim verified (2026-05-18 web fetch):
    "the NDCG@10 boosts from 48.69 to 54.40" — +5.71 NDCG@10 from rerank.

Pipeline: retriever top-100 page → cross-encoder pointwise score → top-K return.

KARARLAR S27-S29 (DR Round B Prompt C cross-validated):
- Primary reranker: nvidia/llama-nemotron-rerank-vl-1b-v2 (1.7B, Eagle SigLIP-2 + Llama-3.2-1B)
  - License: NVIDIA Open Model License Agreement — commercial use TBD
  - Turkish: NOT explicit in card; "multilingual" tag + MIRACL eval (validation required)
- Fallback: Qwen3-VL-8B-Instruct zero-shot pointwise prompt
  - License: Apache-2.0
  - Turkish: native (Qwen3 family TurkBench ~72-77)

Serving:
- vLLM A100 80GB: --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 64
- Latency ~1.2-1.8s per 100 candidate pages (Nemotron reranker)
- Latency ~6-12s per 100 pages (Qwen3-VL-8B fallback)

Refs:
- agent1_prompt2 + agent2_prompt2 (DR Round B Prompt C)
- arXiv:2602.03992 Nemotron ColEmbed V2 §5.2
- https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2 (verified live 2026-05-18)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

RERANK_RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results" / "stage4_rerank"

NEMOTRON_RERANKER_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"
QWEN_RERANKER_ID = "Qwen/Qwen3-VL-8B-Instruct"

QWEN_RERANK_PROMPT_TR = """Aşağıdaki Türkçe kullanıcı sorgusunu ve belge sayfası görüntüsünü dikkatlice incele. Belge sayfası sorguyu cevaplıyor mu?

Sorgu: {query}

Sadece bir sayı döndür:
0 = Sayfa sorguyla ilgisiz
1 = Sayfa kısmen ilgili (cevabın bir kısmı var)
2 = Sayfa tam ilgili (cevap doğrudan sayfada)

Çıktı:"""


@dataclass(frozen=True)
class RerankCandidate:
    page_id: str
    image: Image.Image
    retriever_score: float


@dataclass(frozen=True)
class RerankResult:
    page_id: str
    rerank_score: float
    retriever_score: float
    final_rank: int


class NemotronReranker:
    """nvidia/llama-nemotron-rerank-vl-1b-v2 cross-encoder.

    Forward signature (per model card): pointwise (query, image) -> logit score.
    Use AutoModel + AutoProcessor with trust_remote_code (Eagle architecture).
    """

    def __init__(self, model_id: str = NEMOTRON_RERANKER_ID, device: str = "cuda:0") -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
        self.device = device

    @torch.no_grad()
    def score(self, query: str, images: list[Image.Image], batch_size: int = 16) -> list[float]:
        """Score query against each image; return logit (higher = more relevant)."""
        all_scores: list[float] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            inputs = self.processor(
                text=[query] * len(chunk),
                images=chunk,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            scores = logits.squeeze(-1).float().cpu().tolist()
            all_scores.extend(scores if isinstance(scores, list) else [scores])
        return all_scores


class QwenZeroShotReranker:
    """Fallback: Qwen3-VL-8B zero-shot relevance classifier.

    Computes expected score E[relevance] = sum(p_i * i) over {0, 1, 2} from token
    logits of the next token after the prompt. Native Turkish, no fine-tune needed.
    """

    LABEL_TOKENS = ("0", "1", "2")

    def __init__(self, model_id: str = QWEN_RERANKER_ID, device: str = "cuda:0") -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
        self.device = device
        self.label_ids = [
            self.processor.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self.LABEL_TOKENS
        ]

    @torch.no_grad()
    def score(self, query: str, images: list[Image.Image], batch_size: int = 4) -> list[float]:
        all_scores: list[float] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": QWEN_RERANK_PROMPT_TR.format(query=query)},
                        ],
                    }
                ]
                for img in chunk
            ]
            texts = [
                self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in messages_batch
            ]
            inputs = self.processor(
                text=texts,
                images=chunk,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            next_logits = outputs.logits[:, -1, :].float()
            label_logits = next_logits[:, self.label_ids]
            probs = torch.softmax(label_logits, dim=-1)
            expected = probs[:, 0] * 0 + probs[:, 1] * 1 + probs[:, 2] * 2
            all_scores.extend(expected.cpu().tolist())
        return all_scores


def rerank(
    query: str,
    candidates: list[RerankCandidate],
    reranker: NemotronReranker | QwenZeroShotReranker,
    rerank_top_n: int = 50,
    return_top_k: int = 10,
) -> list[RerankResult]:
    """Top-100 retriever output → top-50 rerank → return top-10.

    Follows Nemotron §5.2 setup (top-50 rerank window).
    """
    sorted_by_retriever = sorted(candidates, key=lambda c: c.retriever_score, reverse=True)
    head = sorted_by_retriever[:rerank_top_n]
    images = [c.image for c in head]
    rerank_scores = reranker.score(query, images)
    scored = sorted(zip(head, rerank_scores), key=lambda x: x[1], reverse=True)
    return [
        RerankResult(
            page_id=cand.page_id,
            rerank_score=float(rs),
            retriever_score=float(cand.retriever_score),
            final_rank=idx + 1,
        )
        for idx, (cand, rs) in enumerate(scored[:return_top_k])
    ]


def load_reranker(backend: Literal["nemotron", "qwen"], device: str = "cuda:0"):
    if backend == "nemotron":
        return NemotronReranker(device=device)
    if backend == "qwen":
        return QwenZeroShotReranker(device=device)
    raise ValueError(f"Unknown backend: {backend!r}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage-4 reranker pipeline")
    parser.add_argument("--backend", choices=["nemotron", "qwen"], default="nemotron")
    parser.add_argument(
        "--validation-set",
        type=Path,
        required=True,
        help="JSONL: {query, candidate_pages: [{page_id, image_path, retriever_score}], gold_page_id}",
    )
    parser.add_argument("--rerank-top-n", type=int, default=50)
    parser.add_argument("--return-top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RERANK_RESULTS_DIR / "rerank_eval.json")
    args = parser.parse_args()

    reranker = load_reranker(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.validation_set.read_text().splitlines() if line.strip()]
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    for row in rows:
        candidates = [
            RerankCandidate(
                page_id=c["page_id"],
                image=Image.open(c["image_path"]).convert("RGB"),
                retriever_score=float(c["retriever_score"]),
            )
            for c in row["candidate_pages"]
        ]
        ranked = rerank(row["query"], candidates, reranker, args.rerank_top_n, args.return_top_k)
        gold = row["gold_page_id"]
        for k in hits:
            if any(r.page_id == gold and r.final_rank <= k for r in ranked):
                hits[k] += 1
        total += 1

    metrics = {f"hit@{k}": v / total for k, v in hits.items()} if total else {}
    args.output.write_text(json.dumps({"backend": args.backend, "total": total, **metrics}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
