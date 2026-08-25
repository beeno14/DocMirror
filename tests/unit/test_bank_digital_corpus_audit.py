from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from docmirror.models.mirror.vnext import EvidenceStore
from docmirror.plugins.bank_statement import statement_context


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tmp" / "pdfs" / "bank_digital_corpus_audit.py"
    spec = importlib.util.spec_from_file_location("bank_digital_corpus_audit_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _terminal_source(field: str) -> dict:
    return {
        "raw_name": f"issuer_{field}",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 3,
                "bbox": [10.0, 500.0, 100.0, 515.0],
            }
        ],
        "evidence_ids": [f"aggregate:{field}"],
    }


def _header() -> dict:
    terminal = {"debit_count": 3, "debit_total": "31.00", "credit_count": 1, "credit_total": "5.00"}
    return {
        "record_id": "statement_header:r000001",
        "normalized": dict(terminal),
        "canonical_raw": {key: str(value) for key, value in terminal.items()},
        "raw": {"本月累计": dict(terminal)},
        "source": {
            "source": "statement_header_scope",
            "page_range": [1, 3],
            "field_sources": {key: _terminal_source(key) for key in terminal},
        },
        "confidence": 1.0,
    }


def _transactions() -> list[dict]:
    values = [
        (1, "expense", "10.00", "90.00"),
        (2, "income", "5.00", "94.00"),
        (3, "expense", "20.00", "74.00"),
    ]
    return [
        {
            "record_id": f"bank:r{index:06d}",
            "normalized": {
                "statement_header_id": "statement_header:r000001",
                "direction": direction,
                "amount": amount,
                "balance": balance,
            },
            "canonical_raw": {"direction": direction, "amount": amount, "balance": balance},
            "raw": {},
            "source": {
                "source": "physical_table",
                "source_page": page,
                "page_range": [page, page],
                "bbox": [10.0, 100.0 + index * 20.0, 400.0, 115.0 + index * 20.0],
                "evidence_ids": [f"row:{index}"],
            },
        }
        for index, (page, direction, amount, balance) in enumerate(values, start=1)
    ]


def _physical_parse_result(*specs: dict) -> SimpleNamespace:
    pages = []
    for spec in specs:
        cell_bbox = spec.get("cell_bbox")
        cell_evidence_ids = list(spec.get("cell_evidence_ids") or [])
        geometry_bbox = spec.get("geometry_bbox", cell_bbox)
        geometry_evidence_ids = list(spec.get("geometry_evidence_ids", cell_evidence_ids) or [])
        cell = SimpleNamespace(bbox=cell_bbox, evidence_ids=cell_evidence_ids)
        row = SimpleNamespace(cells=[cell], source_row_index=0)
        table = SimpleNamespace(
            table_id=spec["table_id"],
            rows=[row],
            metadata={
                "raw_rows": [["value"]],
                "geometry": {
                    "cell_bboxes": [[geometry_bbox]],
                    "cell_evidence_ids": [[geometry_evidence_ids]],
                },
            },
        )
        pages.append(
            SimpleNamespace(
                page_number=spec["page"],
                source_page_number=spec["page"],
                tables=[table],
            )
        )
    return SimpleNamespace(pages=pages)


def _cell_ref(page: int, table_id: str, *, row: int = 0, raw_row: int = 0) -> dict:
    return {"page": page, "table_id": table_id, "row": row, "raw_row": raw_row, "col": 0}


def _reconciled_payload(monkeypatch) -> dict:
    def fact(page: int, value: str) -> statement_context._HeaderFact:
        return statement_context._HeaderFact(
            "brought_forward_balance",
            "承前余额",
            value,
            value,
            page,
            f"page:{page:04d}",
            (20.0, 60.0, 120.0, 75.0),
            (f"carry:{page}",),
        )

    monkeypatch.setattr(
        statement_context,
        "_page_header_facts",
        lambda _result: ({1: [], 2: [fact(2, "89.00")], 3: [fact(3, "94.00")]}, {}),
    )
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {
            "expected_rows": 3,
            "source": "positioned_date_anchors",
            "confidence": 0.80,
            "row_sources": [{"source_page": page} for page in (1, 2, 3)],
            "pages": [1, 2, 3],
        },
    )
    transactions = _transactions()
    [header] = statement_context.reconcile_source_unitemized_residuals(
        SimpleNamespace(),
        transactions,
        [_header()],
        source_route="digital",
        selected_source="canonical_table",
    )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_corpus_audit_accepts_exact_source_unitemized_reconciliation(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "pass"
    assert "header_direction_count_mismatch" not in audit["finding_counts"]
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]
    assert "header_source_unitemized_reconciliation_invalid" not in audit["finding_counts"]
    assert len(payload["datasets"][1]["rows"]) == 3


def test_corpus_audit_rejects_recoverable_physical_provenance_loss(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_bbox_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1


def test_corpus_audit_accepts_source_that_truly_lacks_physical_provenance(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "geo_table_0"
    source["source_cell_refs"] = [_cell_ref(1, "geo_table_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "geo_table_0",
            "cell_bbox": None,
            "cell_evidence_ids": [],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_checks_exact_raw_row_geometry(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": None,
            "cell_evidence_ids": [],
            "geometry_bbox": [10.0, 100.0, 20.0, 110.0],
            "geometry_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_bbox_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1


def test_corpus_audit_accepts_stitched_row_local_source_refs(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["page_range"] = [1, 2]
    source["source_cell_refs"] = [
        _cell_ref(1, "pt_1_0"),
        _cell_ref(2, "pt_2_0"),
    ]
    source["source_refs"] = [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": [10.0, 100.0, 20.0, 110.0],
            "evidence_ids": ["ev:0001:text:000001"],
        },
        {
            "source_page": 2,
            "page_range": [2, 2],
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "evidence_ids": ["ev:0002:text:000001"],
        },
    ]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        },
        {
            "page": 2,
            "table_id": "pt_2_0",
            "cell_bbox": [10.0, 10.0, 20.0, 20.0],
            "cell_evidence_ids": ["ev:0002:text:000001"],
        },
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_does_not_guess_unresolved_physical_refs(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0", row=99, raw_row=99)]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_recovers_strict_native_alias_from_one_exact_public_raw_row(
    monkeypatch,
    audit_module,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "value"}
    source = row["source"]
    source.pop("evidence_ids")
    source["table_id"] = "native:p1:t0"
    # Native row indexes are not proof of cached row identity.  The exact full
    # public raw row below must be what resolves this source.
    source["source_row_index"] = 99
    # This is the same physical row in a quarter-turned top-left coordinate
    # system.  The gate should preserve/accept the native bbox, not compare it
    # numerically with or replace it from the cached physical table.
    source["bbox"] = [15.0, 76.0, 827.0, 102.0]
    source["source_refs"] = [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": list(source["bbox"]),
            "source": "native_pdf_words",
        }
    ]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [493.0, 15.0, 519.0, 827.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_source_cell_refs_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]


def test_corpus_audit_accepts_native_top_level_source_with_exact_pt_cell_refs(
    monkeypatch,
    audit_module,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "value"}
    source = row["source"]
    source["table_id"] = "native:p1:t0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_source_cell_refs_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


@pytest.mark.parametrize("invalid_route", ["raw_mismatch", "duplicate_raw", "noncanonical_table_index"])
def test_corpus_audit_native_alias_fallback_fails_closed(
    monkeypatch,
    audit_module,
    invalid_route,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "different" if invalid_route == "raw_mismatch" else "value"}
    source = row["source"]
    source.pop("evidence_ids")
    source.pop("bbox")
    source["source_refs"] = []
    source["table_id"] = "native:p1:t1" if invalid_route == "noncanonical_table_index" else "native:p1:t0"
    parse_result = _physical_parse_result(
        {
            "page": 1,
            # pt_1_1 at enumerated index zero must not satisfy native ...:t1.
            "table_id": "pt_1_1" if invalid_route == "noncanonical_table_index" else "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )
    if invalid_route == "duplicate_raw":
        table = parse_result.pages[0].tables[0]
        table.metadata["raw_rows"].append(["value"])
        table.metadata["geometry"]["cell_bboxes"].append([[10.0, 120.0, 20.0, 130.0]])
        table.metadata["geometry"]["cell_evidence_ids"].append([["ev:0001:text:000002"]])

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert "recoverable_row_source_cell_refs_missing" not in audit["finding_counts"]
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_rejects_unproven_source_unitemized_reconciliation(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    header = payload["datasets"][0]["rows"][0]
    header["source"]["field_sources"]["source_unitemized_debit_count"].pop("derivation")

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1
    assert audit["finding_counts"]["header_direction_count_mismatch"] == 1


def test_corpus_audit_rejects_source_unitemized_amount_that_does_not_reconcile(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["normalized"]["source_unitemized_debit_amount"] = "2.00"

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def test_corpus_audit_rejects_source_unitemized_field_in_issuer_raw(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["raw"]["source_unitemized_debit_count"] = 1

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def test_corpus_audit_rejects_source_unitemized_field_in_canonical_raw(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["canonical_raw"]["source_unitemized_debit_count"] = 1

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def _period_source(field: str) -> dict:
    return {
        "raw_name": field,
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [10.0, 10.0, 100.0, 20.0],
            }
        ],
        "evidence_ids": [f"period:{field}"],
    }


def _period_payload(
    header_normalized: dict,
    transaction_normalized: list[dict],
    *,
    propagate: bool = False,
    source_only: bool = False,
) -> dict:
    period_fields = {
        field: value
        for field in ("period_start", "period_end")
        if (value := header_normalized.get(field))
    }
    header_field_sources = {field: _period_source(field) for field in period_fields}
    header = {
        "record_id": "statement_header:r000001",
        "normalized": dict(header_normalized),
        "canonical_raw": dict(header_normalized),
        "raw": {},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
            "field_sources": header_field_sources,
        },
    }
    transactions = []
    for index, values in enumerate(transaction_normalized, start=1):
        normalized = {"statement_header_id": header["record_id"], **values}
        canonical = {
            field: values[field]
            for field in ("period_start", "period_end")
            if field in values
        }
        field_sources = {}
        if propagate and not source_only:
            normalized.update(period_fields)
            canonical.update(period_fields)
        if propagate or source_only:
            field_sources = {
                field: dict(header_field_sources[field])
                for field in period_fields
            }
        transactions.append(
            {
                "record_id": f"bank:r{index:06d}",
                "normalized": normalized,
                "canonical_raw": canonical,
                "raw": {},
                "source": {
                    "source": "physical_table",
                    "source_page": 1,
                    "page_range": [1, 1],
                    "bbox": [10.0, 40.0 + index * 10.0, 100.0, 45.0 + index * 10.0],
                    "evidence_ids": [f"row:{index}"],
                    "field_sources": field_sources,
                },
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_period_audit_preserves_coherent_inclusive_header_context(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2023-03-10"}, {"date": "2023-09-10"}],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]
    assert "header_context_not_conserved" not in audit["finding_counts"]


@pytest.mark.parametrize(
    "transaction",
    [
        {"date": "2023-03-09", "period_start": "2023-03-10"},
        {"date": "2023-09-11", "period_end": "2023-09-10"},
    ],
)
def test_period_audit_rejects_transaction_own_open_bound_exclusion(audit_module, transaction):
    payload = _period_payload({}, [transaction])

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_date_outside_own_period"] == 1


def test_period_audit_allows_stripped_context_for_one_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}, {"date": "2023-03-10"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["severity_counts"] == {"diagnostic": 1}
    assert "header_context_not_conserved" not in audit["finding_counts"]


def test_period_audit_rejects_values_propagated_from_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["finding_counts"]["incoherent_header_period_context_propagated"] == 2
    assert audit["finding_counts"]["transaction_date_outside_own_period"] == 1


def test_period_audit_rejects_source_only_propagation_from_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}],
        source_only=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["incoherent_header_period_context_propagated"] == 2
    assert "transaction_date_outside_own_period" not in audit["finding_counts"]


@pytest.mark.parametrize(
    ("header_period", "dates"),
    [
        ({"period_start": "2023-03-10"}, ["2023-03-10", "2023-09-11"]),
        ({"period_end": "2023-09-10"}, ["2023-03-09", "2023-09-10"]),
    ],
)
def test_period_audit_preserves_coherent_one_sided_header_context(
    audit_module,
    header_period,
    dates,
):
    payload = _period_payload(
        header_period,
        [{"date": date_value} for date_value in dates],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]


def test_period_audit_does_not_treat_cutoff_as_a_period_bound(audit_module):
    payload = _period_payload(
        {"statement_cutoff_date": "2023-03-10"},
        [{"date": "2022-09-11"}, {"date": "2023-09-11"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]
    assert "transaction_date_outside_own_period" not in audit["finding_counts"]


def test_period_audit_uses_timestamp_when_normalized_date_is_absent(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10"},
        [{"timestamp": "2023-03-09T23:59:59"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["findings"][0]["detail"]["before_start_rows"] == 1


def _native_text_atom(
    evidence_id: str,
    text: str,
    bbox: list[float],
    *,
    source_kind: str = "pdf_native",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=evidence_id,
        text=text,
        bbox=bbox,
        page_id="page:0001",
        source_kind=source_kind,
        confidence=1.0,
    )


def _balance_header_atoms(*, include_right_neighbor: bool = True) -> list[SimpleNamespace]:
    atoms = [
        _native_text_atom("header:amount", "交易金额", [258.0, 80.0, 290.0, 88.0]),
        _native_text_atom("header:balance:en", "Balance", [337.0, 70.0, 365.0, 78.0]),
        _native_text_atom("header:balance:zh", "余额", [337.0, 80.0, 353.0, 88.0]),
    ]
    if include_right_neighbor:
        atoms.extend(
            [
                _native_text_atom("header:counter", "对方账号", [411.0, 80.0, 443.0, 88.0]),
                _native_text_atom("header:summary", "摘要", [667.0, 80.0, 683.0, 88.0]),
            ]
        )
    return atoms


def _balance_parse_result(
    row_atoms: list[SimpleNamespace],
    *,
    include_right_neighbor: bool = True,
) -> SimpleNamespace:
    atoms = [*_balance_header_atoms(include_right_neighbor=include_right_neighbor), *row_atoms]
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(
            evidence=[SimpleNamespace(text_atoms=atoms)],
        ),
    )


def _evidence_store_parse_results(
    atoms: list[SimpleNamespace],
) -> tuple[SimpleNamespace, dict]:
    serialized_atoms = [dict(vars(atom)) for atom in atoms]
    store = EvidenceStore.model_validate({"text_atoms": serialized_atoms})
    typed = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(evidence=store),
    )
    serialized = {
        "pages": [{"page_number": 1, "source_page_number": 1, "tables": []}],
        "evidence_plane": {"evidence": {"text_atoms": serialized_atoms}},
    }
    return typed, serialized


def _balance_payload(*rows: dict) -> dict:
    header_id = "statement_header:r000001"
    header = {
        "record_id": header_id,
        "normalized": {},
        "canonical_raw": {},
        "raw": {},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
        },
    }
    transactions = []
    for index, spec in enumerate(rows, start=1):
        normalized = {
            "statement_header_id": header_id,
            "balance": spec["normalized_balance"],
            **spec.get("normalized", {}),
        }
        canonical = {
            "balance": spec["canonical_balance"],
            **spec.get("canonical", {}),
        }
        source = {
            "source": "canonical_evidence_table",
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": spec["bbox"],
            "evidence_ids": list(spec["evidence_ids"]),
        }
        if "reconstruction_repairs" in spec:
            source["reconstruction_repairs"] = spec["reconstruction_repairs"]
        transactions.append(
            {
                "record_id": f"records:r{index:06d}",
                "normalized": normalized,
                "canonical_raw": canonical,
                "raw": {"余额": spec["canonical_balance"]},
                "source": source,
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_balance_source_role_gate_rejects_normalized_mismatch(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom]),
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1
    [finding] = [
        item
        for item in audit["findings"]
        if item["code"] == "transaction_balance_source_role_mismatch"
    ]
    assert finding["path"] == "datasets.transactions.rows.0.normalized.balance"
    assert finding["detail"]["reasons"] == ["normalized_value_mismatch"]
    assert finding["detail"]["source_balance"]["evidence_id"] == "row:balance"


def test_balance_source_role_gate_rejects_cross_column_canonical_residue(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "51,492.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    summary_atom = _native_text_atom(
        "row:summary",
        "6.17两单6030",
        [667.0, 120.0, 715.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "51492.51",
            "canonical_balance": "51,492.516.17",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id, summary_atom.id],
            "reconstruction_repairs": [
                {
                    "field": "余额",
                    "ocr_raw": "51,492.516.17",
                    "reconstructed": "51,492.51",
                }
            ],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom, summary_atom]),
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1
    [finding] = [
        item
        for item in audit["findings"]
        if item["code"] == "transaction_balance_source_role_mismatch"
    ]
    assert finding["path"] == "datasets.transactions.rows.0.canonical_raw.balance"
    assert finding["detail"]["reasons"] == ["cross_column_residue"]
    assert finding["detail"]["cross_column_sources"] == [
        {
            "evidence_id": "row:summary",
            "text": "6.17两单6030",
            "bbox": [667.0, 120.0, 715.0, 128.0],
            "role": "summary",
            "residue": "6.17",
        }
    ]


def test_balance_source_role_gate_accepts_numeric_summary_kept_in_its_column(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "51,492.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    summary_atom = _native_text_atom(
        "row:summary",
        "6.17两单6030",
        [667.0, 120.0, 715.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "51492.51",
            "canonical_balance": "51,492.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id, summary_atom.id],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom, summary_atom]),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


@pytest.mark.parametrize("ambiguity", ["missing_neighbor", "multiple_candidates", "ocr_atom"])
def test_balance_source_role_gate_skips_ambiguous_evidence(audit_module, ambiguity):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
        source_kind="ocr" if ambiguity == "ocr_atom" else "pdf_native",
    )
    row_atoms = [balance_atom]
    evidence_ids = [balance_atom.id]
    if ambiguity == "multiple_candidates":
        duplicate = _native_text_atom(
            "row:balance:duplicate",
            "45,462.52",
            [334.0, 130.0, 370.0, 138.0],
        )
        row_atoms.append(duplicate)
        evidence_ids.append(duplicate.id)
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 145.0],
            "evidence_ids": evidence_ids,
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result(
            row_atoms,
            include_right_neighbor=ambiguity != "missing_neighbor",
        ),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


def test_balance_source_role_gate_does_not_hard_fail_filtered_ledger_gap(audit_module):
    first_balance = _native_text_atom(
        "row:balance:1",
        "100.00",
        [337.0, 120.0, 365.0, 128.0],
    )
    second_balance = _native_text_atom(
        "row:balance:2",
        "50.00",
        [337.0, 150.0, 365.0, 158.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "100.00",
            "canonical_balance": "100.00",
            "normalized": {"direction": "expense", "amount": "1.00"},
            "canonical": {"direction": "expense", "amount": "1.00"},
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [first_balance.id],
        },
        {
            "normalized_balance": "50.00",
            "canonical_balance": "50.00",
            "normalized": {"direction": "expense", "amount": "1.00"},
            "canonical": {"direction": "expense", "amount": "1.00"},
            "bbox": [52.0, 145.0, 731.0, 175.0],
            "evidence_ids": [second_balance.id],
        },
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([first_balance, second_balance]),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


def test_balance_source_role_gate_has_typed_evidence_store_mapping_parity(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    atoms = [*_balance_header_atoms(), balance_atom]
    typed, serialized = _evidence_store_parse_results(atoms)
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id],
        }
    )

    typed_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=typed,
    )
    serialized_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=serialized,
    )

    assert audit_module._native_text_atoms(typed) == audit_module._native_text_atoms(serialized)
    assert typed_audit == serialized_audit
    assert typed_audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1


def test_fused_direction_amount_header_does_not_widen_balance_band(audit_module):
    header_atoms = [
        _native_text_atom("header:summary", "摘要", [131.0, 80.0, 150.0, 88.0]),
        _native_text_atom("header:amount", "⽀/收交易⾦额", [208.0, 80.0, 270.0, 88.0]),
        _native_text_atom("header:balance", "账户余额", [280.0, 80.0, 319.0, 88.0]),
        _native_text_atom("header:location", "交易地点", [329.0, 80.0, 368.0, 88.0]),
        _native_text_atom("header:party", "对⽅户名", [410.0, 80.0, 449.0, 88.0]),
    ]
    amount_atom = _native_text_atom("row:amount", "-30,000.00", [210.0, 120.0, 258.0, 128.0])
    fused_balance = _native_text_atom(
        "row:balance-location",
        "170,015.87兴业银行漳州高新",
        [280.0, 120.0, 407.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "170015.87",
            "canonical_balance": "170,015.87",
            "bbox": [29.0, 110.0, 560.0, 140.0],
            "evidence_ids": [amount_atom.id, fused_balance.id],
        }
    )
    typed, serialized = _evidence_store_parse_results(
        [*header_atoms, amount_atom, fused_balance]
    )

    typed_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=typed,
    )
    serialized_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=serialized,
    )

    assert audit_module._source_header_role("⽀/收交易⾦额") == "amount"
    assert typed_audit == serialized_audit
    assert typed_audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in typed_audit["finding_counts"]


def _cross_row_role_parse_result(*, include_crossing_account: bool = True) -> SimpleNamespace:
    atoms = [
        _native_text_atom("header:amount", "交易金额", [208.0, 80.0, 270.0, 88.0]),
        _native_text_atom("header:balance", "账户余额", [280.0, 80.0, 319.0, 88.0]),
        _native_text_atom("header:location", "交易地点", [329.0, 80.0, 368.0, 88.0]),
        _native_text_atom("header:party", "对方户名", [394.0, 80.0, 432.0, 88.0]),
        _native_text_atom("header:account", "对方账户/对方银行", [469.0, 80.0, 550.0, 88.0]),
        _native_text_atom("row:previous:balance", "100.00", [280.0, 116.0, 319.0, 124.0]),
        _native_text_atom("row:previous:party", "支付宝(中国)", [394.0, 140.0, 460.0, 148.0]),
        _native_text_atom("row:following:balance", "75.00", [280.0, 152.0, 319.0, 160.0]),
        _native_text_atom("row:following:party", "网络技术有限公司", [394.0, 152.0, 460.0, 160.0]),
        _native_text_atom("row:following:account", "支付宝(中国)网络技术有限公司", [469.0, 152.0, 565.0, 160.0]),
    ]
    if include_crossing_account:
        atoms.append(
            _native_text_atom("row:previous:account", "215500690", [469.0, 140.0, 518.0, 148.0])
        )
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(evidence=[SimpleNamespace(text_atoms=atoms)]),
    )


def _cross_row_role_payload(*, include_crossing_account: bool = True) -> dict:
    previous_ids = ["row:previous:balance", "row:previous:party"]
    if include_crossing_account:
        previous_ids.append("row:previous:account")
    return _balance_payload(
        {
            "normalized_balance": "100.00",
            "canonical_balance": "100.00",
            "bbox": [29.0, 108.0, 565.0, 150.0],
            "evidence_ids": previous_ids,
        },
        {
            "normalized_balance": "75.00",
            "canonical_balance": "75.00",
            "bbox": [29.0, 146.0, 565.0, 170.0],
            "evidence_ids": [
                "row:following:balance",
                "row:following:party",
                "row:following:account",
            ],
        },
    )


def test_cross_row_source_role_gate_requires_two_independent_columns(audit_module):
    failing = audit_module.audit_community_payload(
        _cross_row_role_payload(),
        effective_page_count=1,
        parse_result=_cross_row_role_parse_result(),
    )
    ambiguous = audit_module.audit_community_payload(
        _cross_row_role_payload(include_crossing_account=False),
        effective_page_count=1,
        parse_result=_cross_row_role_parse_result(include_crossing_account=False),
    )

    assert failing["finding_counts"]["transaction_cross_row_source_role_ownership_mismatch"] == 1
    assert failing["findings"][-1]["path"] == "datasets.transactions.rows.0.source.evidence_ids"
    assert "transaction_cross_row_source_role_ownership_mismatch" not in ambiguous["finding_counts"]


def _direction_distribution_payload(
    rows: list[tuple[str, str]],
    *,
    debit_count: int,
    debit_total: str | None,
    credit_count: int,
    credit_total: str | None,
) -> dict:
    header_id = "statement_header:r000001"
    terminal = {
        "debit_count": debit_count,
        "debit_total": debit_total,
        "credit_count": credit_count,
        "credit_total": credit_total,
        "total_transactions": len(rows),
    }
    header = {
        "record_id": header_id,
        "normalized": dict(terminal),
        "canonical_raw": {key: str(value) for key, value in terminal.items()},
        "raw": {"合计": {key: str(value) for key, value in terminal.items()}},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
        },
    }
    transactions = []
    balance = 1000
    for index, (direction, amount) in enumerate(rows, start=1):
        transactions.append(
            {
                "record_id": f"records:r{index:06d}",
                "normalized": {
                    "statement_header_id": header_id,
                    "direction": direction,
                    "amount": amount,
                    "balance": str(balance),
                },
                "canonical_raw": {
                    "direction": direction,
                    "amount": amount,
                    "balance": str(balance),
                },
                "raw": {},
                "source": {
                    "source": "physical_table",
                    "source_page": 1,
                    "page_range": [1, 1],
                    "bbox": [10.0, 100.0 + index * 20.0, 400.0, 115.0 + index * 20.0],
                    "evidence_ids": [f"row:{index}"],
                },
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


@pytest.mark.parametrize(
    ("debit_count", "debit_total", "credit_count", "credit_total"),
    [
        (2, "30.00", 1, "5.00"),
        (1, "5.00", 2, "30.00"),
    ],
    ids=["direct", "swapped"],
)
def test_header_direction_distribution_accepts_exact_direct_or_swapped_mapping(
    audit_module,
    debit_count,
    debit_total,
    credit_count,
    credit_total,
):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=debit_count,
        debit_total=debit_total,
        credit_count=credit_count,
        credit_total=credit_total,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_header_direction_distribution_rejects_when_neither_mapping_reconciles(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="29.00",
        credit_count=1,
        credit_total="6.00",
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_direction_distribution_mismatch"] == 1
    [finding] = [
        item for item in audit["findings"] if item["code"] == "header_direction_distribution_mismatch"
    ]
    assert finding["detail"]["accepted_mappings"] == []
    assert finding["detail"]["actual_expense_total"] == "30.00"
    assert finding["detail"]["actual_income_total"] == "5.00"


def test_header_direction_distribution_accepts_exact_ambiguous_mapping(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "10.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="10.00",
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_header_direction_distribution_accepts_count_only_direct_mapping(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total=None,
        credit_count=1,
        credit_total=None,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_direction_distribution_accepts_swapped_mapping_with_declared_side_residual(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="5.00",
        credit_count=3,
        credit_total="31.00",
    )

    _, candidates = audit_module._direction_distribution_candidates(
        payload["datasets"][1]["rows"],
        declared_counts={"debit": 1, "credit": 3},
        declared_amounts={"debit": Decimal("5.00"), "credit": Decimal("31.00")},
        residual_counts={"debit": 0, "credit": 1},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("1.00")},
    )

    assert candidates == ("swapped",)


@pytest.mark.parametrize(
    ("rows", "counts", "amounts", "expected"),
    [
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": Decimal("30.00"), "credit": Decimal("5.00")},
            ("direct",),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 1, "credit": 2},
            {"debit": Decimal("5.00"), "credit": Decimal("30.00")},
            ("swapped",),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": Decimal("29.00"), "credit": Decimal("6.00")},
            (),
        ),
        (
            [("expense", "10.00"), ("income", "10.00")],
            {"debit": 1, "credit": 1},
            {"debit": Decimal("10.00"), "credit": Decimal("10.00")},
            ("direct", "swapped"),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": None, "credit": None},
            ("direct",),
        ),
    ],
    ids=["direct-only", "swapped-only", "neither", "ambiguous-both", "count-only"],
)
def test_direction_distribution_candidate_truth_table(
    audit_module,
    rows,
    counts,
    amounts,
    expected,
):
    payload = _direction_distribution_payload(
        rows,
        debit_count=counts["debit"],
        debit_total=None,
        credit_count=counts["credit"],
        credit_total=None,
    )
    transaction_rows = payload["datasets"][1]["rows"]

    _, candidates = audit_module._direction_distribution_candidates(
        transaction_rows,
        declared_counts=counts,
        declared_amounts=amounts,
        residual_counts={"debit": 0, "credit": 0},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("0")},
    )

    assert candidates == expected
