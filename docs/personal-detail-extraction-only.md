# Personal-detail extraction-only hook

Create a dedicated plugin instance to enable the prototype:

```python
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin

reader = CreditReportPlugin(unvalidated=True)
bundle = reader.project_bundle(sealed_parse_result)
payload = bundle.json_payload()
```

`CreditReportPlugin()` and `CreditReportPlugin(unvalidated=False)` keep the
existing checked pipeline. The option is instance-local; it does not mutate
the registered plugin singleton. Enterprise and personal-brief routing is
unchanged. The normal sealed-input integrity guard remains active.

The on-path reads initial OCR/text, key-value pairs and physical table cells.
It reuses the PBOC section/account-family and business-field dictionaries and
the native OCR row grouping. It extracts exact labels, column values, card
blocks and explicit year/month grids. It does not run Candidate B repair,
re-OCR, reconciliation, field validation, population checks or acceptance
gates. It does not read the PDF again.

The output uses the ordinary Community JSON envelope, PBOC dataset names,
record IDs, `normalized`, `raw`, `canonical_raw` and source-page references.
The hook notice is logged, never placed in JSON. Both public and semantic JSON
omit validation/review/confidence/completeness metadata and control datasets.
Business statuses (including repayment, account, case and payment status, and
administrative-review results) remain business data and are retained.

`normalized` here means field naming and presentation conversion, **not a
validity guarantee**. Unparseable values remain strings. Missing table cells
do not move later values into their columns. Duplicate source records remain
separate. Monthly values have only the observed calendar coordinates; missing
status values are not filled from neighboring months. A decoded orphan grid
can be emitted without an account ID.

This is an accuracy-inspection prototype, not a replacement for the checked
pipeline. It cannot recover information missing from initial OCR or guess
unrecognized labels/sections. Units with no field binding are counted in the
log; absent datasets do not assert document absence or successful extraction.
Do not infer completeness from the lack of validation fields. In particular,
free-form narrative and malformed/unregistered layouts can remain unbound.

The focused test module `tests/unit/test_personal_detail_unvalidated.py` uses
only hand-built, report-like tables and OCR lines, including low-confidence
and malformed values. No real reports or PDF/OCR execution are needed.

## Verification (2026-08-27)

- 73 new extraction-only cases passed, including headerless continuation,
  stacked sections, blank columns, malformed scalars/geometry, instance
  isolation, sealed-input integrity and both JSON serialization paths.
- The final focused regression run passed **3,350 tests in 89 files** (34.60s),
  including those 73 cases. Ruff passed for both changed source files and the
  new test file. No source changes were made after this run started.
- Real customer PDFs and saved-report replay modules were excluded. Existing
  unit tests may create small synthetic files in pytest temporary directories.
  These results do not measure real-document extraction accuracy.

Reproduce the selected focused regression run from the repository root:

```powershell
$personalTests = @(Get-ChildItem tests/unit/test_personal_detail*.py |
    Where-Object { $_.Name -notmatch '_(hong|huang|lin|wang|yang|ye|yu|primary)_|replay|corpus' })
$sharedTests = @(Get-ChildItem tests/unit/test_credit_report*.py)
$testPaths = @($personalTests.FullName) + @($sharedTests.FullName)
python -m pytest @testPaths -q --tb=short
```
