# ADR 0003: Business-facing digital-bank Community delivery

- Status: Accepted (user requested)
- Date: 2026-08-27

Digital-bank Community delivery uses v5 while the internal semantic source
retains all extracted normalized values, raw evidence and provenance. v3 and v4
remain valid legacy contracts; all other provider defaults are unchanged.

Dataset rows contain `normalized` business fields followed by compact
`extraction` metadata. Generated record/header relationships move out of the
normalized business-value plane. Source-only business values are promoted from
the internal `additional_fields` array to named, source-labelled columns with
lossless JSON values. No existing normalized business value is overwritten.

Enhanced Markdown renders only business facts and metadata, then a short
extraction appendix. Generated IDs, parser internals and the supplemental-field
wrapper are not rendered. Shared account context is displayed once, without
dropping it from JSON. Full-value display and source redactions are preserved.

CSV business columns follow the new view. Audit CSV and internal source evidence
keep their existing roles. Public replay must preserve sparse/native values and
must not recreate withheld raw evidence. Versioned schema validation and
independent value/identity conservation checks protect the delivery boundary;
extraction strategies and lazy deployment are untouched.
