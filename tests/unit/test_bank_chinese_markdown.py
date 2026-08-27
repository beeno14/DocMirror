"""Chinese presentation must not translate source text or change business facts."""

from __future__ import annotations

import copy
import json
import re

import pytest

from docmirror.output.bank_business_view import render_business_markdown
from docmirror.plugins.bank_statement.community_plugin import BANK_DATA_DICTIONARY, BANK_STANDARD_FIELDS
from docmirror.server.edition_outputs import _write_community_bundle_files
from scripts.validate.bank_business_exports import _MARKDOWN_ENUM_LABELS, assert_business_markdown_values
from scripts.validate.bank_consumer_exports import without_extraction
from tests.unit.test_bank_consumer_output import _consumer_bundle


def _payload():
    return {
        "document": {"title": "银行流水", "type": "bank_statement", "page_count": 1},
        "sections": [],
        "datasets": [{
            "id": "ds_transactions", "name": "transactions", "label": "transactions", "row_count": 2,
            "columns": [{"key": key, "label": key} for key in ("sequence_no", "direction", "amount")],
            "rows": [
                {"normalized": {"sequence_no": "001", "direction": "income", "amount": 20}},
                {"normalized": {"sequence_no": "002", "direction": "expense", "amount": 10}},
            ],
        }],
        "extraction": {"warnings": []},
    }


def test_direction_cells_and_generated_labels_are_chinese_without_mutating_json():
    payload = _payload()
    before = copy.deepcopy(payload)
    text = render_business_markdown(payload)
    assert "## 交易明细" in text
    assert "| 序号 | 收支方向 | 交易金额 |" in text
    assert "| 001 | 收入 | 20 |" in text
    assert "| 002 | 支出 | 10 |" in text
    assert "income" not in text and "expense" not in text
    assert_business_markdown_values(payload, text)
    assert payload == before


def test_only_generated_document_type_titles_are_translated():
    payload = _payload()
    payload["document"]["title"] = "bank_statement"
    assert "# 银行流水" in render_business_markdown(payload)
    payload["document"]["title"] = "Original English Bank Statement"
    assert "# Original English Bank Statement" in render_business_markdown(payload)


@pytest.mark.parametrize(("key", "value", "expected"), [
    ("currency", "CNY", "人民币"), ("currency", "RMB", "人民币"),
    ("currency", "USD", "美元"), ("currency", "HKD", "港元"),
    ("currency", "EUR", "欧元"), ("currency", "JPY", "日元"),
    ("direction_filter", "income", "收入"), ("direction_filter", "expense", "支出"),
    ("direction_filter", "all", "全部"), ("sort_order", "asc", "升序"),
    ("sort_order", "ascending", "升序"), ("sort_order", "desc", "降序"),
    ("sort_order", "descending", "降序"), ("document_type", "bank_reconciliation", "银行对账单"),
])
def test_known_values_translate_in_headers_scalars_and_grouped_facts(key, value, expected):
    payload = _payload()
    payload["sections"] = [{"items": [{"key": key, "value": value}], "groups": [
        {"items": [{"key": key, "value": value, "additional_values": [{"page": 1, "value": value}]}]},
    ]}]
    payload["datasets"].insert(0, {
        "id": "ds_statement_header", "name": "statement_header", "row_count": 1,
        "columns": [{"key": key}], "rows": [{"normalized": {key: value}}],
    })
    before = copy.deepcopy(payload)
    text = render_business_markdown(payload)
    assert f"| {key} |" not in text
    assert f"| {expected} |" in text
    assert f"第 1 页：{expected}" in text
    assert_business_markdown_values(payload, text)
    assert payload == before


def test_currency_translation_covers_shared_context_and_per_transaction_values():
    payload = _payload()
    dataset = payload["datasets"][0]
    dataset["columns"].append({"key": "currency"})
    for row in dataset["rows"]:
        row["normalized"]["currency"] = "CNY"
    text = render_business_markdown(payload)
    assert "**币种:** 人民币" in text and text.count("人民币") == 1
    assert_business_markdown_values(payload, text)
    dataset["rows"][1]["normalized"]["currency"] = "USD"
    text = render_business_markdown(payload)
    assert "| 001 | 收入 | 20 | 人民币 |" in text
    assert "| 002 | 支出 | 10 | 美元 |" in text
    assert_business_markdown_values(payload, text)


def test_source_variants_unknown_values_names_and_nested_free_text_remain_literal():
    payload = _payload()
    dataset = payload["datasets"][0]
    source_columns = [
        {"key": "direction_original", "source_header": "收支原文", "canonical_field": "direction", "label": "收支原文"},
        {"key": "currency_original", "source_header": "Currency", "canonical_field": "currency", "label": "Currency"},
        {"key": "summary", "label": "摘要"}, {"key": "counter_party", "label": "对方户名"},
        {"key": "business_detail", "label": "业务明细"},
    ]
    dataset["columns"].extend(source_columns)
    dataset["rows"][0]["normalized"].update(
        direction_original="income", currency_original="USD", summary="expense / income",
        counter_party="Income Expense LLC", business_detail={"direction": "expense", "currency": "CNY", "确认": False},
    )
    dataset["rows"][1]["normalized"]["direction"] = "bank-private-code"
    before = copy.deepcopy(payload)
    text = render_business_markdown(payload)
    assert "| Currency |" in text
    for source in ("income", "USD", "expense / income", "Income Expense LLC", "direction：expense", "currency：CNY", "false", "bank-private-code"):
        assert source in text
    assert_business_markdown_values(payload, text)
    assert payload == before


def test_synthetic_original_labels_translate_without_losing_collision_suffix_or_source_value():
    payload = _payload()
    dataset = payload["datasets"][0]
    dataset["columns"].append({"key": "direction_original_2", "canonical_field": "direction", "source_header": "direction",
                               "source_occurrence": 2, "label": "direction（原文）（2）"})
    dataset["rows"][0]["normalized"]["direction_original_2"] = "income"
    text = render_business_markdown(payload)
    assert "收支方向（原文）（2）" in text
    assert "| 001 | 收入 | 20 | income |" in text


def test_dictionary_enum_references_work_but_source_columns_never_use_them():
    payload = _payload()
    dataset = payload["datasets"][0]
    dataset["columns"].extend([
        {"key": "flow", "label": "收支", "enum_ref": "direction"},
        {"key": "source_flow", "label": "收支原文", "enum_ref": "direction", "source_header": "原文收支"},
    ])
    dataset["rows"][0]["normalized"].update(flow="expense", source_flow="expense")
    text = render_business_markdown(payload)
    assert "| 001 | 收入 | 20 | 支出 | expense |" in text
    assert_business_markdown_values(payload, text)


@pytest.mark.parametrize(("code", "message", "expected"), [
    ("CQF_DEGRADED", "cqf_degraded:canonical_quality", "规范记录质量检查提示降级，请复核。"),
    ("CQF_LOW_COVERAGE", "cqf_low_coverage:canonical_quality", "规范记录覆盖率偏低，请复核。"),
    ("LOW_COVERAGE", "low_coverage:bank_ledger", "银行流水交易覆盖率偏低，请复核。"),
    ("BANK_PHYSICAL_LOGICAL_ROW_MISMATCH", "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical=18:canonical=19",
     "来源物理行数 18，规范交易笔数 19，请复核跨行或跨页记录。"),
    ("DATASET_COMPLETENESS_UNVERIFIED", "dataset ds_transactions has 19 emitted records but no independent source count",
     "交易明细已输出 19 条记录，但缺少独立的来源笔数，尚不能确认完整性。"),
    ("DATASET_ROW_COUNT_MISMATCH", "dataset ds_transactions emitted 19 of 20 expected records",
     "交易明细预期 20 条记录，实际输出 19 条，请复核。"),
    ("DATASET_VERIFICATION_BLOCKED", "dataset ds_transactions emitted the expected 19 records but domain quality did not permit verification",
     "交易明细已输出预期的 19 条记录，但业务质量检查尚未通过。"),
])
def test_known_generated_warnings_are_chinese_and_keep_counts_without_changing_json(code, message, expected):
    payload = _payload()
    payload["extraction"]["warnings"] = [{"code": code, "message": message}]
    before = copy.deepcopy(payload)
    text = render_business_markdown(payload)
    appendix = text.split("## 提取说明", 1)[1]
    assert expected in appendix
    assert message not in appendix
    assert code not in appendix
    assert payload == before


@pytest.mark.parametrize("code", ["CQF_DEGRADED", "NEW_DIAGNOSTIC"])
def test_unknown_warning_formats_keep_the_entire_message(code):
    payload = _payload()
    payload["extraction"]["warnings"] = [{"code": code, "message": "new diagnostic details: 0123"}]
    text = render_business_markdown(payload)
    assert "new diagnostic details: 0123" in text


@pytest.mark.parametrize("bad_cells", [("支出", "收入"), ("收入", "收入"), ("income", "expense")])
def test_independent_auditor_catches_swapped_wrong_or_untranslated_directions(bad_cells):
    payload = _payload()
    # Both words remain elsewhere, so whole-document string presence is not sufficient.
    payload["sections"] = [{"items": [{"key": "note", "value": "收入 支出 income expense"}], "groups": []}]
    text = render_business_markdown(payload)
    broken = text.replace("| 001 | 收入 |", f"| 001 | {bad_cells[0]} |").replace("| 002 | 支出 |", f"| 002 | {bad_cells[1]} |")
    with pytest.raises(AssertionError, match="transaction directions"):
        assert_business_markdown_values(payload, broken)


def test_bank_dictionary_covers_standard_fields_legacy_names_and_independent_enum_contract():
    assert set(BANK_STANDARD_FIELDS) <= BANK_DATA_DICTIONARY["record_columns"].keys()
    for pool in (BANK_DATA_DICTIONARY["fields"], BANK_DATA_DICTIONARY["record_columns"],
                 BANK_DATA_DICTIONARY["datasets"]["statement_header"]["columns"]):
        assert all(not re.search(r"[A-Za-z]", spec["label"]) for spec in pool.values())
    for key in ("account_name", "statement_period", "total_deposits", "total_withdrawals", "transaction_count"):
        assert BANK_DATA_DICTIONARY["fields"][key]["label"]
    for key, labels in _MARKDOWN_ENUM_LABELS.items():
        assert BANK_DATA_DICTIONARY["enums"][key] == labels


def test_both_production_markdowns_and_cached_payload_use_chinese_with_identical_json_csv_and_evidence(tmp_path):
    bundle, sealed = _consumer_bundle()
    before = copy.deepcopy(bundle.semantic_payload())
    public = bundle.json_payload()
    csvs = bundle.render_dataset_csvs()
    with without_extraction():
        paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=bundle.document["id"])
    text = paths["content"].read_text(encoding="utf-8")
    assert text == paths["enhanced_reading"].read_text(encoding="utf-8")
    assert text == bundle.render_enhanced_markdown(public_payload=public)
    assert "| income |" not in text and "| expense |" not in text
    assert_business_markdown_values(public, text)
    assert public == json.loads(paths["community"].read_text(encoding="utf-8"))
    assert before == bundle.semantic_payload()
    assert csvs == bundle.render_dataset_csvs()
    assert sealed.verify_integrity()
