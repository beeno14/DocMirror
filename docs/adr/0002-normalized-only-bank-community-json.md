# ADR 0002: Normalized-only digital-bank Community JSON

- Status: Accepted (user requested)
- Date: 2026-08-27

Digital-bank Community delivery uses version 4.0.0 and a single normalized
business-value plane. It omits record `raw` / `canonical_raw` and scalar-item
`raw`. This supersedes the source-pool delivery requirement of ADR 0001 only
for this opt-in provider route. Other providers and scanned bank output remain
v3, and both contracts are validated explicitly.

The extraction result and internal semantic source retain their raw evidence.
Before delivery, source business fields not demonstrably represented in the
standard normalized fields are preserved in `normalized.additional_fields`.
No original normalized value is overwritten and no extraction rule is added.
Audits use the actual retained evidence, never a reconstruction of raw values
from the normalized-only JSON. Record/count/order conservation is unchanged.

Consumers of v4 use `normalized` plus the dataset column catalog. They can no
longer assume that source-format round-tripping is available from the delivery
JSON alone. Source-faithful Markdown and internal audit evidence retain their
separate review role.

Digital-bank enhanced Markdown shows original values without application-added
masking. The explicit full-value display policy is persisted in the public
reading plan so JSON replay cannot silently re-mask identifiers. Redactions
already present in source data are retained, not reconstructed.
