# Compact digital-bank Community output

## Current delivery: business-facing v5

Digital-bank delivery now uses schema **5.0.0**. The v4 description below
documents the retained internal source-accounting stage and supported legacy
delivery format; it is no longer the default digital-bank presentation.

Each v5 dataset row puts business values in `normalized` and small technical
metadata in a trailing `extraction` object. That object contains the stable
record ID, page range, optional statement-header relationship, confidence and
review metadata. Full field provenance, candidate/repair details, raw pools and
parser traces remain internal. Dataset keys and relationships use the new
`extraction.record_id` / `extraction.statement_header_id` paths.

The internal `additional_fields` array is expanded into individually named
business columns. Original source headings are retained once in the column
catalog as `source_header`; duplicate headings and key collisions are
disambiguated without replacing existing normalized values. Source variants of
a canonical value have an explicit original-value label. The `json` column
type preserves mixed native values, including strings with leading zeros,
numbers, false, null, lists and objects. Absent schema-only columns disappear
from the v5 catalog; sparse promoted fields stay sparse on replay.

Both delivered Markdown files (`content.md` and `enhanced_reading.md`) show the
same business facts, account information, and transaction
tables. Common account context is shown once above the transaction table.
Generated IDs, the supplemental-field wrapper, and parser diagnostics are not
rendered in the business body. A short trailing extraction appendix preserves
count-verification status and warnings. Values remain unmasked. Dataset CSVs
follow the business columns; the internal audit CSV is unchanged.
Generated counterparty-presence statuses, repeated page-number labels, and
explicit page-local income/expense amounts and counts are not consumer fields.
Whole-statement totals (including existing sums of explicit page totals),
transaction amounts/directions, source transaction columns and unknown business
fields remain unchanged. Page labels are matched exactly in header context;
the filter never searches transaction cell contents. Internal source evidence
and `render_source_markdown()` retain the original page summaries.
The v5 catalog omits raw/evidence-availability flags and redundant storage-path
metadata. Older v5 deliveries are cleaned on replay; v3/v4 contracts and other
providers retain their original defaults.
Source markup is displayed literally, not interpreted as HTML or Markdown.
Within table cells, `↵` represents an original line break; lists use separators.
The standalone artifact validator checks v5 identities and HTML-free Markdown
as well as the existing v3/v4 contracts.

This is a delivery-only adapter, not an extraction or reconstruction strategy.
Only the digital-bank plugin opts in. Other providers, scanned bank statements,
and legacy v3/v4 validation remain supported without changing their defaults.

Saved, audited v4 results can be refreshed without extraction using
`scripts.validate.bank_compact_exports --business-report <report> --artifact-dir
<new-directory> --output <new-report>`. Refresh and replay independently check
the original normalized values, every promoted source value, row identities,
relationships, metadata, Markdown coverage and CSV business fields.

## Earlier compact and normalized-only contracts

Digital bank statements opt into compact Community formatting through the
provider's semantic `compact_output` extension. Other providers, scanned bank
statements, Mirror, Enterprise, and Finance retain their existing defaults.
This is not a new public request selector and does not alter extraction.

## JSON and enhanced Markdown

Optional normalized fields absent from every source row in a dataset may be
omitted from its JSON records. The dataset lists those keys in
`omitted_normalized_fields`. Its complete existing `columns` catalog and CSV columns
remain available. Consumers should use optional-key access for normalized
fields rather than assume that every declared column appears in every record.

The decision uses the original projected records, before serialization fills
schema columns. Any nonempty value (including zero and false), explicit null,
canonical source key, field-level evidence, required field, foreign key, or
matching source heading preserves the field. A source heading printed with
blank cells is retained. Unrecognized blank source headings conservatively
disable omission for that dataset. Source-label matching reuses existing bank
aliases; it does not extract, infer, or repair business values.

Enhanced Markdown and the JSON reading plan omit the same absent columns.
Reopening compact JSON through the compatibility adapter preserves its sparse
rows and compact writing. If an omitted field subsequently acquires a value or
field-level evidence, stale omission metadata cannot hide it.
Source-faithful `content.md`, internal `raw` and `canonical_raw`, source evidence,
warnings, record IDs, row order, counts, and completeness metadata are unchanged.
No provenance pooling or internal evidence deletion is performed.

Opted-in Community JSON is written without indentation. This changes only
whitespace outside JSON strings; values and Unicode contents round-trip
unchanged. Other providers keep their existing pretty-printed JSON.

## Normalized-only v4 delivery

Digital bank output now declares `schema.version = "4.0.0"`. Dataset records
contain only `normalized` as their business-value plane, alongside record IDs,
source references, confidence, and review metadata. The v4 schema rejects
record-level `raw` and `canonical_raw`; scalar section/group items omit `raw`
too. Other pipelines and scanned-bank statements continue producing v3, whose
source-pool requirements are still enforced. The schema registry accepts both.

The complete internal semantic result retains both source pools for auditing.
The normalized-only projection is never used to fabricate missing raw evidence.
Replaying a v4 JSON preserves its native structured values and sparse fields,
but does not recreate source values that are deliberately absent from delivery.

### Supplemental business fields

Existing normalized values are never overwritten. Before source pools are
removed, each source field is accounted for against its declared canonical
role. Exact values and unambiguous number/date formatting conversions need no
duplicate. Unknown labels, compound cells, distinct source values, conflicting
values, and unrepresented canonical values are conservatively retained in the
optional `normalized.additional_fields` array:

```json
{
  "name": "核心流水号",
  "value": "0000123400567890123456"
}
```

An entry may also specify `field` when an existing canonical role is known.
Digital-bank enhanced Markdown displays all values without application-added
masking, including account/customer/identity fields and supplemental values.
The explicit `reading.privacy_mode: "full"` policy survives Community JSON
replay. Source-redacted values remain exactly as supplied; missing digits are
never invented. Other providers and scanned-bank defaults are unchanged.
Values retain their source types and leading zeros; an unrelated field with
the same value is not evidence that a source column was represented. This is
an extensible business schema, not another wholesale copy of the raw pools.
It does not invent extraction mappings or resolve ambiguous source semantics.

Datasets with supplemental fields declare an `array` column. Enhanced Markdown
and dataset CSV expose that normalized column; CSV encodes it as JSON. Existing
CSV columns and audit cells retain their values. Scalar items use
`additional_values` if their source representation contains information not
represented by `value`. Internal audit CSVs retain source evidence; replay from
normalized-only JSON leaves unavailable raw audit cells blank.

### Validation

Focused cases cover source coverage, long identifiers, compound values,
unknown/blank headers, zero versus false, structured values, schema versions,
replay, CSV, and deliberately corrupted exports. Primary and Secondary runs
compare extraction against frozen evidence-backed outputs, independently account
for all source fields, and audit the normalized export against a hash-checked
internal evidence snapshot. Those snapshots are private harness artifacts, not
additional delivery JSON files. Extraction strategies and lazy routing are
unchanged.

Presentation-only changes can be validated from those snapshots with
`scripts.validate.bank_compact_exports --unmask-report <report> --artifact-dir
<new-directory> --output <new-report>`, followed by `--replay-report <new-report>
--output <audit-report>`. This does not execute extraction, does not overwrite
the original artifacts, and allows only the explicit full-value reading policy
to change in JSON. It rechecks source-field coverage and current renderers.
