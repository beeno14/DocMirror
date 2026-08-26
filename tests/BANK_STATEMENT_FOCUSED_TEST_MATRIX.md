# Bank statement focused test matrix

This catalog covers the fast, synthetic bank-statement contract suite. It is test-only: real PDFs, PDF parsing, corpus extraction, and corpus reprojection are explicitly excluded. Numeric corpus signatures below are traceability labels for minimized fixtures, not PDF dependencies.

Status legend: **covered** means the owning test already existed or was previously exercised; **constructed / not executed** marks the latest additions from the current test-construction-only phase.

## Parser strategies

| Strategy | Owning test modules | Positive contracts | Adversarial contracts |
|---|---|---|---|
| Grid standard | `unit/test_bank_institution_grid.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_statement_registry.py`, `unit/test_canonical_quality.py` | Resolve institution-shaped grids; retain split debit/credit amounts, balances, business columns, headers, and source cells. | Reject repeated headers as rows, fused or shifted columns, cross-column balance residue, weak institution hints, and incomplete canonical rows. |
| Compact merged | `unit/test_bank_compact_parser.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_statement_registry.py` | Split compact date/money/balance signatures and preserve the reconstructed row contract. | Fail closed on ambiguous numeric fusion, invalid date or money boundaries, shifted balances, and partial compact signatures. |
| Signed amount | `unit/test_bank_styles_signed_amount.py`, `unit/test_bank_statement_header_projection.py` | Convert a signed source amount into magnitude plus direction while retaining the signed canonical value and promoted business fields. | Reject mixed or incoherent sign evidence, unsafe header promotion, balance-chain contradictions, and rows without bounded source evidence. |
| Borderless / positioned evidence | `unit/test_bank_borderless_native_recovery.py`, `unit/test_bank_evidence_atom_table_recovery.py`, `unit/test_bank_extract_pipeline.py` | Reconstruct columns and wrapped rows from positioned native/OCR atoms with stable row and cell provenance. | Reject column drift, touching-header fusion, ambiguous row ownership, missing atom identity, geometry gaps, and source-role mismatch. |

## Six provider shapes

These are provider-shaped fixtures; they do not authorize issuer inference from filenames.

| Provider shape | Owning test modules | Positive contracts | Adversarial contracts |
|---|---|---|---|
| Agricultural Bank current-account detail | `unit/test_bank_identity_extraction.py`, `unit/test_bank_statement_context.py`, `unit/test_bank_institution_grid.py` | Compatibility-glyph identity, explicit account/holder/currency, split periods, and debit/credit grids. | Counterparty/body identity, routing-only institution hints, and malformed account labels cannot become statement identity. |
| Construction Bank dense primary/duplicate sequence | `unit/test_bank_statement_domain_solver.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | Source-bounded primary row census agrees with the duplicate sequence and terminal evidence. | Missing terminal tuples, deleted duplicate ordinals, reordered pages, or asymmetric row loss invalidate exactness. |
| China Merchants primary/duplicate sequence | `unit/test_bank_statement_domain_solver.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | Complete primary and duplicate transaction signatures agree on row count and terminal row. | Primary-only or duplicate-only terminal loss, ambiguous anchors, and disagreement remain non-authoritative. |
| Bank of Communications wide monthly reconciliation | `unit/test_bank_statement_context.py`, `unit/test_bank_institution_grid.py` | Page/month metadata, wide cumulative totals, opening-branch currency, page labels, and adjacent reconciliation seal code remain source-backed. | Distant/ambiguous cumulative cells, body carry values, page furniture, and seal-like codes without one nearby stamp fail closed. |
| Ping An OCR monthly statement | `unit/test_bank_statement_context.py`, `unit/test_bank_borderless_native_recovery.py` | OCR fallback preserves monthly scopes, statement number, brought-forward values, print metadata, and page provenance. | Native/OCR mixing, cross-scope carry, unbounded month text, and conflicting scope identity are rejected. |
| Payment-certificate / digital-wallet statement | `unit/test_bank_statement_context.py` | Embedded subject identity, currency/unit, and exact transaction period are promoted only from the certificate structure. | Invalid identity numbers, prose lookalikes, counterparty labels, and contaminated packed fields cannot become statement facts. |

## Deployment and arbitration

| Branch | Owning test modules | Positive contracts | Adversarial contracts |
|---|---|---|---|
| Style registry and strategy dispatch | `unit/test_bank_statement_registry.py`, `unit/test_bank_extract_pipeline.py` | A supported detected style selects its registered parser and returns its canonical records. | Unsupported, low-confidence, empty, or failed candidates take only the declared fallback/degraded path. |
| Candidate quality and row accounting | `unit/test_canonical_quality.py`, `unit/test_bank_blo.py`, `unit/test_bank_extract_pipeline.py` | Complete candidate coverage, amount consistency, and balance ordering support selection. | Duplicate rows, accounting disagreement, unsafe reordering, low coverage, and canonical/source count mismatch block promotion. |
| Issuer row-count authority | `unit/test_bank_statement_domain_solver.py`, `unit/test_bank_digital_corpus_audit.py` | Exact counts publish only from complete, ordered, issuer-specific source evidence. | Generic labels, nonterminal totals, page gaps, duplicate pages, and conflicting count sources remain non-authoritative. |
| Native, OCR, and evidence-table arbitration | `unit/test_bank_evidence_atom_table_recovery.py`, `unit/test_bank_borderless_native_recovery.py`, `unit/test_bank_extract_pipeline.py` | The selected source owns every emitted row and its evidence chain. | Anchor-dependent self-validation, mixed-source rows, missing cell owners, and ambiguous geometry cannot certify completeness. |

## Identity, context, and Community projection

| Layer | Owning test modules | Positive contracts | Adversarial contracts |
|---|---|---|---|
| Identity | `unit/test_bank_identity_extraction.py` | Chinese and English labelled single periods, leap-year year/month evidence, coherent multi-page envelopes, exact source components, and explicit issuer/account identity. | Overlap, gap, reorder, duplicate-page, mixed-scope, distant geometry, missing atom IDs, filename-only issuer, and ledger/counterparty pollution fail closed. |
| Statement context | `unit/test_bank_statement_context.py` | Multiple scopes, substantive query filters, page-local and terminal aggregates, titles, disclaimers, page labels, seals, stable context attachment, and scope-isolated carry-forward residuals. | Ledger labels and wrapped body cells never become headers; incomplete aggregates, ambiguous geometry, weak provenance, carry contradictions, and cross-scope leakage are rejected. |
| Community projection | `unit/test_bank_statement_header_projection.py` | `statement_header` precedes `transactions`; foreign keys attach the correct scope; raw/canonical/normalized/source values survive; normalized-only period envelopes expose exact raw components without inventing raw `query_period`; promoted signed rows retain business fields and source. | Filename/routing issuer hints cannot publish business identity, source rows are not overwritten by context, and normalized-only data cannot be backfilled into raw pools. |

Latest additions in `unit/test_bank_statement_context.py` are **constructed / not executed** in this phase:

- `test_english_page_reset_keeps_source_headers_and_ledger_body_scope_local`
- `test_english_page_label_requires_one_tightly_bounded_source_value`
- `test_english_disclaimer_is_source_business_text_but_never_a_statement_title`

## Provenance and conservation

| Contract | Owning test modules | Positive contracts | Adversarial contracts |
|---|---|---|---|
| Evidence provenance | `unit/test_bank_evidence_atom_table_recovery.py`, `unit/test_bank_statement_context.py`, `unit/test_bank_styles_signed_amount.py` | Evidence IDs, page/table/row/cell ownership, bounding boxes, source refs, derivations, and component lists remain attached to the fact they support. | Missing IDs, distant or ambiguous atoms, cross-column refs, conflicting owners, and inferred-only details cannot claim source authority. |
| Raw/canonical/normalized conservation | `unit/test_bank_statement_header_projection.py`, `unit/test_community_bundle_derived_record_pools.py` | Source raw labels, canonical source strings, normalized values, and normalized-only derivations occupy their correct pools. | Bundle projection must not fabricate raw values, erase signed canonical amounts, or lose exact source components. |
| Audit gates | `unit/test_bank_digital_corpus_audit.py`, `unit/test_canonical_quality.py` | Exact row/header/transaction totals and source-role checks pass for internally coherent synthetic payloads. | Balance-role mismatch, cross-column residue, missing derivation, invented raw fields, row-count disagreement, and provenance ambiguity produce findings or fail exactness. |

## Real-derived compact signatures

| Traceability label | Owning test modules | Positive contract | Adversarial contract |
|---|---|---|---|
| Compact 25 | `unit/test_bank_compact_parser.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | The minimized compact date/amount/balance signature reconstructs one exact canonical row with source ownership. | Numeric fusion or field-boundary mutation cannot silently produce a different valid row. |
| Compact 109 | `unit/test_bank_compact_parser.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | The minimized signature retains its exact business tuple and raw-to-canonical mapping. | Truncation, shifted balance, or ambiguous segmentation fails closed. |
| Compact 125 | `unit/test_bank_compact_parser.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | The minimized signature preserves row count, monetary precision, and evidence-backed field boundaries. | Competing splits or incomplete source anchors cannot be promoted. |
| Compact 132 | `unit/test_bank_compact_parser.py`, `unit/test_bank_extract_pipeline.py`, `unit/test_bank_digital_corpus_audit.py` | The minimized signature remains stable through extraction and audit projection. | Duplicate or cross-row residue cannot satisfy conservation. |
| Promoted signed | `unit/test_bank_styles_signed_amount.py`, `unit/test_bank_statement_header_projection.py` | A promoted row preserves `业务类型 → transaction_name`, `票据号 → voucher_number`, signed canonical amount plus normalized magnitude/direction, and `对手户名 → counter_party`, including source provenance. | Promotion without a bounded source row, coherent sign, or complete business mapping is rejected.

No entry in this matrix reads, parses, renders, or audits a real PDF.
