"""Consumer cleanup must not remove transaction/statement facts or audit evidence."""

from __future__ import annotations

import copy
import csv
import io
import json

import pytest

from docmirror.models.entities.parse_result import ParseResult
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.output.bank_business_view import business_view
from docmirror.output.community_bundle import CommunityDataset
from docmirror.server.edition_outputs import _write_community_bundle_files
from docmirror.server.output_builder import materialize_community_bundle
from scripts.validate.bank_business_exports import (
    assert_business_markdown_values,
    assert_business_value_conservation,
    evidence_delivery,
)
from scripts.validate.bank_compact_exports import bind_internal_evidence, validate_and_write_bank_exports
from tests.unit.test_community_bank_business_view import _business_bundle
from tests.unit.test_community_compact_output import _column, _normalized_bundle


def _consumer_bundle():
    """Simulate a two-page bank statement with explicit footer totals and IDs."""
    bundle, sealed = _business_bundle()
    transactions = bundle.datasets[0]
    transactions.public["columns"].append(_column("counterparty_status", label="对方信息状态"))
    template = copy.deepcopy(transactions.rows[0])
    transactions.rows = []
    for index, (page, amount, direction) in enumerate(
        [(1, 10, "expense"), (1, 2, "income"), (2, 20, "expense"), (2, 3, "income"), (2, 0, "income")], 1
    ):
        row = copy.deepcopy(template)
        row["record_id"] = f"records:r{index:06d}"
        row["normalized"].update(amount=amount, direction=direction)
        source_amount = f"{-amount if direction == 'expense' else amount:.2f}"
        row["raw"]["交易金额"] = row["canonical_raw"]["amount"] = source_amount
        row["source"]["page_range"] = [page, page]
        row["normalized"]["counterparty_status"] = "source_null" if not row["normalized"].get("counter_party") else "present"
        transactions.rows.append(row)
    transactions.public["completeness"] = {"expected_row_count": 5, "emitted_row_count": 5, "omitted_row_count": 0,
                                           "verified": True, "basis": "source_count"}
    bundle.document["page_count"] = 2
    bundle.sections[0]["page_range"] = [1, 2]
    public = copy.deepcopy(transactions.public)
    public.update(id="ds_statement_header", name="statement_header", label="流水表头", grain="statement",
                  csv="001_datasets/statement_header.csv", columns=[
                      _column("account_number", label="账号"),
                      _column("debit_total", label="借方总金额", type="money"),
                      _column("credit_total", label="贷方总金额", type="money"),
                      _column("total_transactions", label="交易总笔数", type="integer"),
                  ])
    public.pop("foreign_keys", None)
    public["completeness"] = {"expected_row_count": 1, "emitted_row_count": 1, "omitted_row_count": 0,
                              "verified": True, "basis": "source_header"}
    header = {
        "record_id": "header:r000001",
        "normalized": {"account_number": "0001234567899876", "debit_total": 30.0, "credit_total": 5.0,
                       "total_transactions": 5},
        "canonical_raw": {"debit_total": "30.00", "credit_total": "5.00"},
        "raw": {
            "本页支出算术合计:": [{"page": 1, "value": "10.00"}, {"page": 2, "value": "20.00"}],
            "本页收入算数合计:": [{"page": 1, "value": "2.00"}, {"page": 2, "value": "3.00"}],
            "本页支出笔数:": [{"page": 1, "value": "1"}, {"page": 2, "value": "1"}],
            "页码：": ["1/2", "2/2"],
            "业务说明": "本页收入合计并不代表全年总收入",
            "全年收入合计": "000005.00",
        },
        "source": {"page_range": [1, 2], "field_sources": {
            "debit_total": {"source": "derived_explicit_page_aggregate", "derivation": "sum_explicit_page_totals"},
            "credit_total": {"source": "derived_explicit_page_aggregate", "derivation": "sum_explicit_page_totals"},
        }},
    }
    bundle.datasets.insert(0, CommunityDataset(public=public, rows=[header]))
    bundle.sections[0]["dataset_refs"].insert(0, "ds_statement_header")
    bundle.content_markdown_override = "# 原始页\n\n本页支出算术合计: 10.00\n\n<!-- original-source -->"
    return bundle, sealed


def test_both_consumer_markdowns_and_json_are_clean_but_statement_totals_and_evidence_survive(tmp_path):
    bundle, sealed = _consumer_bundle()
    before = copy.deepcopy((bundle.datasets, bundle.sections))
    evidence = bundle.semantic_payload()
    original = evidence_delivery(evidence)
    public, report = validate_and_write_bank_exports(bundle, tmp_path)
    assert_business_value_conservation(original, public)
    assert bind_internal_evidence(public, evidence)["datasets"][0]["rows"] == evidence["datasets"][0]["rows"]
    assert (bundle.datasets, bundle.sections) == before
    assert sealed.verify_integrity()
    header, transactions = public["datasets"]
    assert header["rows"][0]["normalized"]["debit_total"] == "30.0"
    assert header["rows"][0]["normalized"]["credit_total"] == "5.0"
    assert header["rows"][0]["normalized"]["total_transactions"] == 5
    for dataset in public["datasets"]:
        assert not {"storage_role", "record_path"}.intersection(dataset)
        for column in dataset["columns"]:
            assert not {"raw_available", "evidence_available"}.intersection(column)
            assert column["key"] != "counterparty_status"
    assert [row["normalized"]["amount"] for row in transactions["rows"]] == [10, 2, 20, 3, 0]
    assert transactions["rows"][0]["normalized"]["approved"] is False
    for kind in ("content", "enhanced_reading"):
        text = (tmp_path / f"001_{kind}.md").read_text(encoding="utf-8")
        assert_business_markdown_values(public, text)
        assert "对方信息状态" not in text
        assert "本页支出算术合计" not in text
        assert "本页收入算数合计" not in text
        assert "本页支出笔数" not in text
        assert "| 页码 |" not in text
        assert "0001234567899876" in text
        assert "本页收入合计并不代表全年总收入" in text
        assert "全年收入合计" in text and "000005.00" in text
    assert bundle.render_markdown() == bundle.render_enhanced_markdown()
    assert "本页支出算术合计" in bundle.render_source_markdown()
    assert report["internal_source_markdown_unchanged"] is True
    assert "本页支出算术合计:" in evidence["datasets"][0]["rows"][0]["raw"]
    assert "counterparty_status" in evidence["datasets"][1]["rows"][0]["normalized"]
    csv_rows = list(csv.DictReader(io.StringIO(bundle.render_dataset_csvs()["001_datasets/statement_header.csv"].lstrip("\ufeff"))))
    assert "本页支出算术合计" not in csv_rows[0]
    assert csv_rows[0]["debit_total"] == "30.0"


@pytest.mark.parametrize("label", [
    "本页支出算数合计:", " 本 页 收 入 算 术 合 计 ： ", "本页借方合计", "本页贷方金额合计",
    "本页支出笔数:", "本页交易笔数", "Page Income Total:", "Page Expense Count", "页码：",
])
@pytest.mark.parametrize("value", [0, False, None, [{"page": 1, "value": "0.00"}]])
def test_page_summary_policy_is_header_scoped_and_not_a_cell_value_heuristic(label, value):
    bundle, _ = _consumer_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    for dataset in original["datasets"]:
        dataset["rows"][0]["normalized"]["additional_fields"] = [{"name": label, "value": value}]
    clean = business_view(original)
    assert_business_value_conservation(original, clean)
    assert not any(column.get("source_header") == label for column in clean["datasets"][0]["columns"])
    column = next(column for column in clean["datasets"][1]["columns"] if column.get("source_header") == label)
    assert clean["datasets"][1]["rows"][0]["normalized"][column["key"]] == value


def test_direct_and_grouped_page_summary_fields_are_omitted_without_hiding_other_scalars():
    bundle, _ = _consumer_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    header = original["datasets"][0]
    header["columns"].append(_column("page_debit", label="本页支出合计"))
    header["rows"][0]["normalized"]["page_debit"] = 77
    original["sections"][0]["items"].append({"key": "page_credit", "label": "本页收入合计", "value": 88, "type": "money"})
    original["sections"][0]["groups"].append({"key": "totals", "label": "汇总", "items": [
        {"key": "page_credit", "label": "本页收入合计", "value": 88, "type": "money"},
        {"key": "credit_total", "label": "收入总额", "value": 100, "type": "money"},
    ]})
    clean = business_view(original)
    assert_business_value_conservation(original, clean)
    assert "page_debit" not in clean["datasets"][0]["rows"][0]["normalized"]
    assert clean["sections"][0]["groups"][-1]["items"] == [
        {"key": "credit_total", "label": "收入总额", "value": 100, "type": "money"},
    ]


def test_literal_source_status_is_preserved_even_when_its_generated_counterpart_is_hidden():
    bundle, _ = _consumer_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    original["datasets"][1]["rows"][0]["normalized"]["additional_fields"] = [
        {"name": "counterparty_status", "field": "counterparty_status", "value": "bank supplied business status"},
    ]
    cleaned = business_view(original)
    assert_business_value_conservation(original, cleaned)
    row = cleaned["datasets"][1]["rows"][0]
    assert "counterparty_status" not in row["normalized"]
    assert row["normalized"]["counterparty_status_original"] == "bank supplied business status"


@pytest.mark.parametrize("field", ["amount", "direction", "debit_total", "credit_total", "total_transactions", "全年收入合计"])
def test_auditor_still_rejects_loss_of_real_transaction_or_statement_facts(field):
    bundle, _ = _consumer_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    clean = bundle.json_payload()
    dataset = clean["datasets"][1 if field in {"amount", "direction"} else 0]
    dataset["rows"][0]["normalized"].pop(field)
    with pytest.raises(AssertionError, match="business row values"):
        assert_business_value_conservation(original, clean)


def test_older_v5_replay_cannot_reintroduce_intermediate_fields_or_page_summaries(tmp_path):
    bundle, _ = _consumer_bundle()
    clean = bundle.json_payload()
    old = copy.deepcopy(clean)
    header, transactions = old["datasets"]
    header["columns"].append(_column("debit_total_original", label="本页支出算术合计", source_header="本页支出算术合计:",
                                     canonical_field="debit_total", source_occurrence=1))
    header["rows"][0]["normalized"]["debit_total_original"] = [{"page": 1, "value": "10.00"}]
    transactions["columns"].append(_column("counterparty_status", label="对方信息状态"))
    for row in transactions["rows"]:
        row["normalized"]["counterparty_status"] = "present"
    for dataset in old["datasets"]:
        dataset.update(storage_role="canonical", record_path="rows")
        for column in dataset["columns"]:
            column.update(raw_available=True, evidence_available=True)
    before = copy.deepcopy(old)
    assert business_view(old) == clean
    assert old == before
    for _ in range(3):
        restored = materialize_community_bundle(old, ParseResult())
        assert restored.json_payload() == clean
        assert restored.render_markdown() == bundle.render_markdown()
        old = restored.json_payload()
    paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=clean["document"]["id"])
    assert json.loads(paths["community"].read_text(encoding="utf-8")) == clean


@pytest.mark.parametrize("corruption", ["counterparty_status", "source_header_page_label", "storage_role", "record_path",
                                        "raw_available", "evidence_available"])
def test_business_schema_rejects_newly_excluded_intermediates(corruption):
    bundle, _ = _consumer_bundle()
    public = bundle.json_payload()
    dataset = public["datasets"][1]
    if corruption in {"storage_role", "record_path"}:
        dataset[corruption] = "canonical" if corruption == "storage_role" else "rows"
    elif corruption in {"raw_available", "evidence_available"}:
        dataset["columns"][0][corruption] = False
    else:
        dataset["rows"][0]["normalized"][corruption] = "internal"
    assert not validate_projection_payload("community", public).valid


def test_legacy_non_opted_in_outputs_keep_source_markdown_and_availability_contracts(tmp_path):
    bundle, _ = _normalized_bundle()
    bundle.content_markdown_override = "# 原始页\n\n本页收入合计: 17.00"
    original = bundle.json_payload()
    assert original["schema"]["version"] == "4.0.0"
    assert "storage_role" in original["datasets"][0]
    assert "raw_available" in original["datasets"][0]["columns"][0]
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=original["document"]["id"])
    assert paths["content"].read_text(encoding="utf-8") == bundle.render_source_markdown()
    broken = copy.deepcopy(original)
    broken["datasets"][0]["columns"][0].pop("raw_available")
    assert not validate_projection_payload("community", broken).valid
