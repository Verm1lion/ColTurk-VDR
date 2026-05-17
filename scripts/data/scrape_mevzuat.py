"""Scrape Resmi Gazete / mevzuat.gov.tr public legal corpus.

Placeholder — Day 2-3 implementation, depends on OPEN_ITEMS.md A5 resolution
(ToS check + rate-limit policy).

Strategy:
- Playwright async scraper with rate-limit (SCRAPE_DELAY_SECONDS env var)
- Save PDFs to data/raw/mevzuat/
- Per-document metadata: title, date, category, source URL, license="public_domain_TR"
- Output: data/processed/mevzuat_corpus.parquet
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Day 2-3 task. See OPEN_ITEMS.md A5.")


if __name__ == "__main__":
    main()
