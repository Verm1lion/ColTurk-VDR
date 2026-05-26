"""Sentetik TR query QC filtresi — promptc.md §2 quality control thresholds.

Mert workflow:
1. Claude.ai web UI'da TR query üret (prompt 04 veya 05)
2. JSONL'leri `data/synthetic_tr/queries/<source>_queries.jsonl` altına kaydet
3. Bu script'i çalıştır → reject etmemiş query'ler `<source>_queries_filtered.jsonl`'a yazılır
4. QC raporu (`<source>_qc_report.json`) reject sayılarını verir

Filtreler (KARARLAR S33 + DR Round B Prompt C §1c):
1. Length: 5-30 word (template-y kısa veya gereksiz uzun reject)
2. TR langdetect ≥ 0.85 (non-Turkish drift reject)
3. Diacritic count ≥ 1 (ASCII-only TR — model dejenerasyonu sinyali; reject)
4. Exact-substring overlap with source page text < 70% (verbatim copy reject)
5. Self-reference: "bu sayfada / yukarıdaki belgede" gibi ifade reject
6. Per-page duplicate (cosine similarity > 0.85) — sadece en kısa olanı tut
7. Generic-ness: top-1% generic-query reservoir cosine > 0.9 → reject

Beklenen reject rate: %5-15. **%50+ ise prompt revize gerekli.**

Input JSONL format:
    {"page_id": "...", "source": "...", "page_text": "..." (optional, for overlap check),
     "queries": ["q1", "q2", "q3"]}

Output JSONL format (only kept queries):
    {"page_id": "...", "source": "...", "queries_kept": ["q1", "q2"], "queries_rejected": [{"query": "...", "reason": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("validate_synthetic_queries")

MIN_WORDS = 5
MAX_WORDS = 30
MIN_DIACRITIC_COUNT = 1
MAX_EXACT_OVERLAP_RATIO = 0.70
LANGDETECT_TR_THRESHOLD = 0.85
PER_PAGE_DUPLICATE_THRESHOLD = 0.85
GENERIC_RESERVOIR_THRESHOLD = 0.90

TURKISH_DIACRITICS = set("ğüşıöçĞÜŞİÖÇ")

SELF_REFERENCE_PATTERNS = [
    r"\bbu sayfada\b",
    r"\byukarıda(ki)?\b",
    r"\başağıda(ki)?\b",
    r"\bverilen belgede\b",
    r"\bbu dokümanda\b",
    r"\bbu rapor(da|un)\b",
    r"\bişbu (sayfa|belge|fatura)\b",
]
SELF_REFERENCE_RE = re.compile("|".join(SELF_REFERENCE_PATTERNS), re.IGNORECASE)

GENERIC_RESERVOIR = [
    "İK politikası nedir?",
    "Finansal durum nasıl?",
    "Şirket hakkında bilgi verir misiniz?",
    "Hangi ürünler var?",
    "Toplam tutar nedir?",
    "Kim imza atmış?",
    "Hangi tarihte yapılmış?",
    "Detayları paylaşabilir misin?",
    "Açıklayabilir misin?",
    "Bu nedir?",
]


@dataclass
class ValidationResult:
    page_id: str
    source: str
    queries_kept: list[str]
    queries_rejected: list[dict[str, str]]


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_diacritics(text: str) -> int:
    return sum(1 for c in text if c in TURKISH_DIACRITICS)


def check_self_reference(text: str) -> bool:
    return bool(SELF_REFERENCE_RE.search(text))


def ngram_overlap_ratio(query: str, source_text: str, n: int = 4) -> float:
    """Token-level n-gram overlap (Jaccard-style) between query and source page text."""
    if not source_text:
        return 0.0
    q_tokens = re.findall(r"\b\w+\b", query.lower(), flags=re.UNICODE)
    s_tokens = re.findall(r"\b\w+\b", source_text.lower(), flags=re.UNICODE)
    if len(q_tokens) < n:
        return 0.0
    q_ngrams = {tuple(q_tokens[i : i + n]) for i in range(len(q_tokens) - n + 1)}
    s_ngrams = {tuple(s_tokens[i : i + n]) for i in range(len(s_tokens) - n + 1)}
    if not q_ngrams:
        return 0.0
    return len(q_ngrams & s_ngrams) / len(q_ngrams)


def detect_tr_score(text: str) -> float:
    """Lightweight TR detection without langdetect dependency.

    Returns proportion of "TR-likely" tokens: Turkish-character or common TR stopword.
    For production, install `langdetect` and replace with `detect_langs(text)`.
    """
    tr_chars = sum(1 for c in text.lower() if c in "ğüşıöç")
    tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    if not tokens:
        return 0.0
    tr_stopwords = {
        "ve", "ile", "için", "olan", "olarak", "bir", "bu", "da", "de", "ki",
        "ne", "mı", "mi", "mu", "mü", "ya", "veya", "ama", "ancak", "fakat",
        "ise", "den", "dan", "den", "den", "kadar", "göre", "üzere", "üzerine",
    }
    tr_stop_count = sum(1 for t in tokens if t in tr_stopwords)
    score = (tr_chars / max(len(text), 1)) * 2 + (tr_stop_count / max(len(tokens), 1))
    return min(score, 1.0)


def cosine_jaccard(q1: str, q2: str) -> float:
    """Cheap Jaccard cosine approximation on token bags."""
    t1 = set(re.findall(r"\b\w+\b", q1.lower(), flags=re.UNICODE))
    t2 = set(re.findall(r"\b\w+\b", q2.lower(), flags=re.UNICODE))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / (len(t1 | t2) ** 0.5 * len(t1) ** 0.5)


def validate_query(query: str, source_text: str | None) -> tuple[bool, str | None]:
    """Return (keep, reject_reason). True iff query passes all filters."""
    if not query or not isinstance(query, str):
        return False, "empty_or_not_string"

    word_count = count_words(query)
    if word_count < MIN_WORDS:
        return False, f"too_short_{word_count}_words"
    if word_count > MAX_WORDS:
        return False, f"too_long_{word_count}_words"

    if count_diacritics(query) < MIN_DIACRITIC_COUNT:
        return False, "no_turkish_diacritics"

    if check_self_reference(query):
        return False, "self_reference"

    if detect_tr_score(query) < LANGDETECT_TR_THRESHOLD * 0.5:
        return False, "low_turkish_score"

    if source_text:
        overlap = ngram_overlap_ratio(query, source_text)
        if overlap > MAX_EXACT_OVERLAP_RATIO:
            return False, f"high_overlap_{overlap:.2f}"

    for generic in GENERIC_RESERVOIR:
        if cosine_jaccard(query, generic) > GENERIC_RESERVOIR_THRESHOLD:
            return False, "too_generic"

    return True, None


def validate_page_queries(
    page_id: str,
    source: str,
    queries: list[str],
    source_text: str | None,
) -> ValidationResult:
    kept: list[str] = []
    rejected: list[dict[str, str]] = []

    for q in queries:
        keep, reason = validate_query(q, source_text)
        if not keep:
            rejected.append({"query": q, "reason": reason or "unknown"})
            continue
        # Per-page duplicate check (against already-kept)
        is_dup = False
        for kept_q in kept:
            if cosine_jaccard(q, kept_q) > PER_PAGE_DUPLICATE_THRESHOLD:
                is_dup = True
                rejected.append({"query": q, "reason": "duplicate_of_kept"})
                break
        if not is_dup:
            kept.append(q)

    return ValidationResult(
        page_id=page_id,
        source=source,
        queries_kept=kept,
        queries_rejected=rejected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentetik TR query QC filtresi")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Filtered JSONL (kept only)")
    parser.add_argument("--qc-report", type=Path, required=True, help="QC JSON report")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    LOG.info("Validating %d rows from %s", len(rows), args.input)

    results: list[ValidationResult] = []
    reject_counter: Counter[str] = Counter()
    total_queries = 0
    total_kept = 0

    for row in rows:
        queries = row.get("queries", [])
        source_text = row.get("page_text")  # optional
        result = validate_page_queries(
            page_id=row["page_id"],
            source=row.get("source", "unknown"),
            queries=queries,
            source_text=source_text,
        )
        results.append(result)
        total_queries += len(queries)
        total_kept += len(result.queries_kept)
        for rej in result.queries_rejected:
            reject_counter[rej["reason"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "page_id": r.page_id,
                        "source": r.source,
                        "queries_kept": r.queries_kept,
                        "queries_rejected": r.queries_rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    reject_rate = (total_queries - total_kept) / max(total_queries, 1)
    qc_summary = {
        "input_file": str(args.input),
        "total_pages": len(rows),
        "total_queries": total_queries,
        "total_kept": total_kept,
        "total_rejected": total_queries - total_kept,
        "reject_rate": round(reject_rate, 4),
        "reject_reasons": dict(reject_counter.most_common()),
        "thresholds": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "min_diacritic_count": MIN_DIACRITIC_COUNT,
            "max_exact_overlap_ratio": MAX_EXACT_OVERLAP_RATIO,
            "langdetect_tr_threshold": LANGDETECT_TR_THRESHOLD,
            "per_page_duplicate_threshold": PER_PAGE_DUPLICATE_THRESHOLD,
            "generic_reservoir_threshold": GENERIC_RESERVOIR_THRESHOLD,
        },
        "verdict": (
            "OK (reject rate 5-15% normal)"
            if 0.05 <= reject_rate <= 0.15
            else "LOW_REJECT — quality may be over-permissive"
            if reject_rate < 0.05
            else "HIGH_REJECT — prompt revize gerekli"
        ),
    }
    args.qc_report.parent.mkdir(parents=True, exist_ok=True)
    args.qc_report.write_text(json.dumps(qc_summary, indent=2, ensure_ascii=False))

    LOG.info(
        "Validation complete: %d/%d kept (%.1f%% rejected). Report: %s",
        total_kept, total_queries, reject_rate * 100, args.qc_report,
    )
    print(json.dumps(qc_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
