# Long-document data for the 32K stage (P4): verified sources

Verified on 2026-10-10 by sampling the Hugging Face repos. Tokens ≈ chars/4, computed from the first shard's head, so all figures are [EXTERNAL]/sample estimates. All repos are ungated.
In Common Pile, the licence is per document (`metadata.license`) and may occasionally be wrong (paper 2506.05209). Common Pile's Comma v0.1 was trained on the same data and released openly, which is a precedent for using it.

| # | Repo | Licence | Size | Long docs |
|---|---|---|---|---|
| 1 | common-pile/project_gutenberg | Public domain (EN) | 71.8K docs, ~6.5B tok | 95% ≥8K, 71% ≥32K |
| 2 | common-pile/pre_1929_books_filtered | Public domain (US) | 125K docs, ~11.5B tok | 91% ≥8K, 70% ≥32K. OCR needs dehyphenation |
| 3 | common-pile/library_of_congress | Public domain (card cc-by-4.0) | 135K docs, ~12B tok | 93% ≥8K, 55% ≥32K. Some OCR noise |
| 4 | common-pile/arxiv_papers | CC-BY / CC-BY-SA / CC0 | 321K docs, ~5.3B tok | 50% ≥8K, 3.8% ≥32K |
| 5 | common-pile/doab | CC-BY / CC-BY-SA | 474K docs, ~3B tok | 8.6% ≥8K, but those hold 83% of tokens. Needs a length filter |
| 6 | common-pile/usgpo | Public domain (US federal) | ~18B tok | 14% ≥8K. Boilerplate-heavy, so cap and dedupe |
| 7 | common-pile/pubmed | CC-BY / CC0 | 159 GB | 37% ≥8K, ~0% ≥32K. Use as 8–16K filler |
| 8 | common-pile/wikimedia | CC-BY-SA 4.0 | 32 GB | Use namespace 0 only |

Code: common-pile/stackv2 (Blue Oak permissive licences, per file). Long code needs repo-level concatenation via `metadata.url`, with data and CSV files excluded.

## Rejected
- **deepmind/pg19:** a loader script; masking applied; already covered by Gutenberg.
- **manu/project_gutenberg:** no licence on the card.
- **dolma3 longmino:** research/education only, and the PDF copyrights are not cleared.
- **the-stack-v2 / starcoderdata:** gated, licence "other".
- **placeholderlabs:** no licence.
- **PleIAs/common_corpus:** fallback only (mixed per-document licences, very large shards).

## Proposed ~1B-token long mix [ESTIMATE]
- **Books 35–40%:** Gutenberg, pre-1929 filtered, LoC.
- **Papers 25–30%:** arXiv ≥12K, DOAB ≥16K, PMC ≥16K.
- **Long gov/wiki ~15%:** USGPO reports, Wikisource and Wikipedia ns0. UK Hansard (common-pile/uk_hansard, Open Parliament Licence v3.0, attribution) was APPROVED by the user on 2026-10-10. Credit it in the release attribution file.
- **Code ~20%:** repo-level Stack v2 concatenations.

## Processing
- Dedupe across the book sources.
- Normalise OCR whitespace and dehyphenate.
- Release needs an attribution file for the CC-BY and CC-BY-SA material.
- Mix 50/50 with replayed normal-length data during the 32K stage.
