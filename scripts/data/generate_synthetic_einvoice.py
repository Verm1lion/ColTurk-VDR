"""Generate synthetic Turkish e-invoice / KYC / receipt PDFs.

Placeholder — Day 3-4, depends on OPEN_ITEMS.md A8 (FATURA TR-render) + A12 (KVKK).

Strategy:
- GİB UBL-TR 1.2.1 XSD schema as template
- Faker-tr for: company names, addresses, TCKN (force-invalidated), IBAN (force-invalidated)
- reportlab + WeasyPrint for PDF render
- Variation: layout templates × N, scan degradation (blur, rotation, JPEG, color jitter)
- KVKK guard: assert TCKN checksum != valid for ALL generated identifiers
- Output: data/processed/synthetic_einvoice/{N}.pdf + metadata.json
- License: MIT (will publish as HF dataset: Verm1ion/turkish-doc-synth-Nk)
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Day 3-4 task. See OPEN_ITEMS.md A8 + A12.")


if __name__ == "__main__":
    main()
