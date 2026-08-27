"""Business-only delivery contracts, using source-like rows and adversarial fields."""

from __future__ import annotations

import copy
import csv
import io
import json
from dataclasses import replace

import pytest

from docmirror.models.entities.parse_result import ParseResult
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.output.bank_business_view import (
    INTERNAL_ROW_FIELDS,
    business_view,
    render_business_markdown,
)
from docmirror.server.edition_outputs import _write_community_bundle_files
from docmirror.server.output_builder import materialize_community_bundle
from scripts.validate.bank_business_exports import (
    assert_business_markdown_values,
    assert_business_value_conservation,
    evidence_delivery,
)
from scripts.validate.bank_compact_exports import bind_internal_evidence, validate_and_write_bank_exports
from scripts.validate.validate_community_artifacts import validate_community_artifacts
from tests.unit.test_community_compact_output import _column, _normalized_bundle


def _business_bundle():
    bundle, sealed = _normalized_bundle()
    bundle.domain["extensions"]["compact_output"]["business_view"] = True
    bundle.domain["extensions"]["enhanced_markdown"] = {"privacy_mode": "full"}
    dataset = bundle.datasets[0]
    dataset.public["columns"].extend([
        _column("statement_header_id", label="流水表头记录ID"),
        _column("own_account", label="本方账号", sensitive=True, display="masked"),
        _column("currency", label="币种"),
        _column("approved", label="确认状态", type="boolean"),
    ])
    row = dataset.rows[0]
    row["normalized"].update(statement_header_id="header:r000001", own_account="0001234567899876",
                             currency="CNY", approved=False, summary="货款|采购\n次行", amount=25)
    row["raw"].update({"核心流水号": "00007777", "交易金额": "-25.00", "业务详情": [{"数量": 0, "确认": False}],
                       "打印代码": '{"code":"0001"}', "摘要": "货款|采购\n次行", "源空字段": ""})
    row["canonical_raw"]["amount"] = "-25.00"
    row["source"].update(bbox=[1, 2, 3, 4], source_refs=[{"table_id": "DEBUG_TABLE_SENTINEL"}],
                         field_sources={"amount": {"raw_name": "交易金额", "candidate": "DEBUG_CANDIDATE_SENTINEL"}})
    row["confidence"] = 0.95
    second = copy.deepcopy(row)
    second["record_id"] = "records:r000002"
    second["normalized"].update(amount=0, approved=True, summary="第二笔", counter_party="示例商户")
    second["raw"].update({"交易金额": "0.00", "摘要": "第二笔", "业务详情": "普通文本", "对方户名": "示例商户"})
    second["raw"].pop("打印代码")
    second["canonical_raw"]["amount"] = "0.00"
    dataset.rows.append(second)
    bundle.sections[0]["items"].extend([
        {"key": "style_id", "label": "版式标识", "value": "DEBUG_STYLE_SENTINEL", "raw": "DEBUG_STYLE_SENTINEL", "type": "string"},
        {"key": "source_reported_transaction_count", "label": "原文报告交易笔数", "value": 2, "raw": "2", "type": "integer"},
    ])
    bundle.warnings.append({"code": "BUSINESS_REVIEW", "level": "warning", "message": "复核提示仅在附录出现"})
    return bundle, sealed


def test_business_view_moves_metadata_preserves_values_and_does_not_mutate_internal_facts() -> None:
    bundle, sealed = _business_bundle()
    before = copy.deepcopy(bundle.datasets)
    semantic = bundle.semantic_payload()
    original = evidence_delivery(semantic)
    payload = bundle.json_payload(semantic)
    assert payload["schema"]["version"] == "5.0.0"
    assert validate_projection_payload("community", payload).valid
    assert validate_projection_payload("community_semantic", semantic).valid
    assert_business_value_conservation(original, payload)
    assert bind_internal_evidence(payload, semantic)["datasets"][0]["rows"] == semantic["datasets"][0]["rows"]
    assert bundle.datasets == before
    assert sealed.verify_integrity()
    assert next(reversed(payload)) == "extraction"
    dataset = payload["datasets"][0]
    assert dataset["primary_key"] == "extraction.record_id"
    assert "omitted_normalized_fields" not in dataset
    assert not INTERNAL_ROW_FIELDS.intersection(column["key"] for column in dataset["columns"])
    for row in dataset["rows"]:
        assert list(row) == ["normalized", "extraction"]
        assert not INTERNAL_ROW_FIELDS.intersection(row["normalized"])
        assert row["extraction"]["statement_header_id"] == "header:r000001"
        assert row["extraction"]["confidence"] == 0.95
        assert "source_refs" not in row["extraction"]
    source_columns = {column["source_header"]: column["key"] for column in dataset["columns"] if "source_header" in column}
    assert dataset["rows"][0]["normalized"][source_columns["核心流水号"]] == "00007777"
    assert dataset["rows"][0]["normalized"][source_columns["交易金额"]] == "-25.00"
    assert dataset["rows"][0]["normalized"][source_columns["打印代码"]] == '{"code":"0001"}'
    assert dataset["rows"][0]["normalized"][source_columns["业务详情"]] == [{"数量": 0, "确认": False}]
    assert dataset["rows"][1]["normalized"][source_columns["业务详情"]] == "普通文本"
    assert "raw" in semantic["datasets"][0]["rows"][0]
    assert "field_sources" in semantic["datasets"][0]["rows"][0]["source"]


def test_business_markdown_is_unmasked_readable_and_puts_only_small_extraction_notes_last(monkeypatch) -> None:
    bundle, _ = _business_bundle()

    def reject_mask(_value):
        raise AssertionError("business Markdown must not add masking")

    monkeypatch.setattr("docmirror.output.community_bundle._masked_display", reject_mask)
    payload = bundle.json_payload()
    markdown = bundle.render_enhanced_markdown()
    assert_business_markdown_values(payload, markdown)
    assert markdown == render_business_markdown(payload)
    body, appendix = markdown.split("\n\n## 提取说明\n\n", 1)
    for text in ("補充業務字段", "补充业务字段", "流水表头记录ID", "statement_header_id",
                 "records:r000001", "header:r000001", "DEBUG_STYLE_SENTINEL", "DEBUG_CANDIDATE_SENTINEL"):
        assert text not in markdown
    assert "核心流水号" in body and "00007777" in body
    assert "0001234567899876" in body
    assert body.count("0001234567899876") == 1  # Shared account context is rendered once.
    assert "货款\\|采购 ↵ 次行" in body
    assert "false" in body and "true" in body
    assert "原文报告交易笔数" in body
    assert "复核提示仅在附录出现" not in body
    assert "复核提示仅在附录出现" in appendix
    assert "尚未核验" in appendix


def test_v5_replay_preserves_sparse_promoted_fields_native_types_csv_and_reading(tmp_path) -> None:
    bundle, _ = _business_bundle()
    payload = bundle.json_payload()
    markdown = bundle.render_enhanced_markdown()
    source_csvs = bundle.render_dataset_csvs()
    for _ in range(3):
        restored = materialize_community_bundle(payload, ParseResult())
        assert restored.json_payload() == payload
        assert restored.render_enhanced_markdown() == markdown
        assert restored.render_dataset_csvs() == source_csvs
        assert restored.conservation_issues(payload=payload, dataset_csvs=source_csvs) == []
        assert all(row["raw"] == row["canonical_raw"] == {} for row in restored.semantic_payload()["datasets"][0]["rows"])
        payload = restored.json_payload()
    paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=payload["document"]["id"])
    assert json.loads(paths["community"].read_text(encoding="utf-8")) == payload
    assert paths["enhanced_reading"].read_text(encoding="utf-8") == markdown
    assert all(row["raw"] == "" for row in csv.DictReader(io.StringIO(restored.render_audit_csv())))


def test_business_production_writer_checks_dense_frozen_values_and_internal_audit_cells(tmp_path) -> None:
    bundle, _ = _business_bundle()
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    baseline = replace(bundle, domain=dense_domain).json_payload()
    payload, report = validate_and_write_bank_exports(bundle, tmp_path, baseline=baseline)
    assert report["business_view"] is True
    assert report["artifact_contract_checked"] is True
    assert report["existing_csv_business_fields_unchanged"] is True
    assert report["existing_audit_cells_unchanged"] is True
    assert report["source_markdown_unchanged"] is False
    assert report["internal_source_markdown_unchanged"] is True
    assert report["compact_json_bytes"] < report["dense_json_bytes"]
    assert payload["extraction"]["warnings"] == baseline["warnings"]


def test_promoted_name_collisions_duplicates_empty_labels_and_formula_like_values_are_lossless() -> None:
    bundle, _ = _business_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    items = [{"name": name, "value": value} for name, value in (
        ("", "0001"), ("", False), ("amount", "-7.00"), ("amount", 0),
        ("additional_fields", [False, 0, None]), ("_page_start", "BUSINESS_PAGE_VALUE"),
        ("record_id", "SOURCE_RECORD_ID"), ("列名", "=SUM(1,2)"),
        ("=SUM(3,4)", "literal header value"),
    )]
    original["datasets"][0]["rows"][0]["normalized"]["additional_fields"] = items
    cleaned = business_view(original)
    assert_business_value_conservation(original, cleaned)
    assert validate_projection_payload("community", cleaned).valid
    keys = [column["key"] for column in cleaned["datasets"][0]["columns"]]
    assert len(keys) == len(set(keys))
    assert "_page_start" not in keys
    assert not any(key.startswith(("=", "+", "-", "@")) for key in keys)
    assert any(column.get("source_header") == "=SUM(3,4)" for column in cleaned["datasets"][0]["columns"])
    restored = materialize_community_bundle(cleaned, ParseResult())
    assert restored.json_payload() == cleaned
    rendered = next(iter(restored.render_dataset_csvs().values()))
    assert "'=SUM(1,2)" in rendered


@pytest.mark.parametrize("corruption", ["amount_type", "promoted_value", "drop_row", "reorder", "page", "warnings", "technical"])
def test_independent_business_auditor_rejects_silent_value_loss_and_metadata_corruption(corruption) -> None:
    bundle, _ = _business_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    broken = bundle.json_payload()
    rows = broken["datasets"][0]["rows"]
    if corruption == "amount_type":
        rows[1]["normalized"]["amount"] = False
    elif corruption == "promoted_value":
        key = next(column["key"] for column in broken["datasets"][0]["columns"] if column.get("source_header") == "核心流水号")
        rows[0]["normalized"].pop(key)
    elif corruption == "drop_row":
        rows.pop()
    elif corruption == "reorder":
        rows.reverse()
    elif corruption == "page":
        rows[0]["extraction"]["page_range"] = [9, 9]
    elif corruption == "warnings":
        broken["extraction"]["warnings"] = []
    else:
        rows[0]["normalized"]["style_id"] = "internal"
    with pytest.raises(AssertionError):
        assert_business_value_conservation(original, broken)


@pytest.mark.parametrize("corruption", ["raw", "source", "additional_fields", "statement_header_id", "style_id", "field_sources", "identity", "masked"])
def test_v5_schema_rejects_intermediate_data_and_lost_unmasked_policy(corruption) -> None:
    bundle, _ = _business_bundle()
    payload = bundle.json_payload()
    row = payload["datasets"][0]["rows"][0]
    if corruption in {"raw", "source"}:
        row[corruption] = {}
    elif corruption in {"additional_fields", "statement_header_id", "style_id"}:
        row["normalized"][corruption] = [] if corruption == "additional_fields" else "internal"
    elif corruption == "field_sources":
        row["extraction"]["field_sources"] = {}
    elif corruption == "identity":
        row["extraction"].pop("record_id")
    else:
        payload["reading"]["privacy_mode"] = "masked"
    assert not validate_projection_payload("community", payload).valid


def test_scalar_source_variants_are_visible_even_when_normalized_value_is_missing() -> None:
    bundle, _ = _business_bundle()
    original = evidence_delivery(bundle.semantic_payload())
    original["sections"][0]["items"].append({"key": "unknown_note", "label": "原文说明", "value": None,
                                             "type": "text", "additional_values": ["必须保留的业务说明"]})
    payload = business_view(original)
    markdown = render_business_markdown(payload)
    assert "必须保留的业务说明" in markdown.split("## 提取说明")[0]
    assert_business_markdown_values(payload, markdown)


def test_business_view_is_opt_in_and_refuses_unaccounted_or_other_provider_payloads() -> None:
    bundle, _ = _normalized_bundle()
    legacy = bundle.json_payload()
    assert legacy["schema"]["version"] == "4.0.0"
    assert "source" in legacy["datasets"][0]["rows"][0]
    assert validate_projection_payload("community", legacy).valid
    for domain, version in (("invoice", "4.0.0"), ("bank_statement", "3.0.0")):
        invalid = copy.deepcopy(legacy)
        invalid["schema"].update(domain=domain, version=version)
        with pytest.raises(ValueError):
            business_view(invalid)


def test_context_only_rows_still_render_each_record_in_the_business_table() -> None:
    bundle, _ = _business_bundle()
    payload = bundle.json_payload()
    dataset = payload["datasets"][0]
    dataset["columns"] = [_column("own_account", label="本方账号")]
    for row in dataset["rows"]:
        row["normalized"] = {"own_account": "0001234567899876"}
    markdown = render_business_markdown(payload)
    assert markdown.count("| 0001234567899876 |") == 2


@pytest.mark.parametrize("replayed", [False, True])
def test_standalone_artifact_validator_accepts_v5_business_delivery(tmp_path, replayed) -> None:
    bundle, _ = _business_bundle()
    if replayed:
        bundle = materialize_community_bundle(bundle.json_payload(), ParseResult())
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=bundle.document["id"])
    assert validate_community_artifacts(paths["community"]) == []


@pytest.mark.parametrize("corruption", [
    "missing_extraction", "invalid_extraction", "missing_normalized", "missing_id",
    "empty_id", "duplicate_id", "reorder", "drop_row", "raw", "source", "root_extraction",
])
def test_standalone_artifact_validator_rejects_broken_v5_delivery(tmp_path, corruption) -> None:
    bundle, _ = _business_bundle()
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=bundle.document["id"])
    payload = json.loads(paths["community"].read_text(encoding="utf-8"))
    rows = payload["datasets"][0]["rows"]
    if corruption == "missing_extraction":
        rows[0].pop("extraction")
    elif corruption == "invalid_extraction":
        rows[0]["extraction"] = None
    elif corruption == "missing_normalized":
        rows[0].pop("normalized")
    elif corruption == "missing_id":
        rows[0]["extraction"].pop("record_id")
    elif corruption == "empty_id":
        rows[0]["extraction"]["record_id"] = ""
    elif corruption == "duplicate_id":
        rows[1]["extraction"]["record_id"] = rows[0]["extraction"]["record_id"]
    elif corruption == "reorder":
        rows.reverse()
    elif corruption == "drop_row":
        rows.pop()
    elif corruption in {"raw", "source"}:
        rows[0][corruption] = {}
    else:
        payload.pop("extraction")
    paths["community"].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    issues = validate_community_artifacts(paths["community"])
    assert issues
    if corruption == "reorder":
        assert any("ordered record_id mismatch" in issue for issue in issues)
    if corruption == "duplicate_id":
        assert any("duplicate record_id" in issue for issue in issues)


@pytest.mark.parametrize(("value", "display"), [
    ("<script>literal source</script>", "&lt;script&gt;literal source&lt;/script&gt;"),
    ("A & B < C", "A &amp; B &lt; C"),
    ("![source](https://example.test/image)", r"!\[source\](https://example.test/image)"),
    ("[source](https://example.test/path)", r"\[source\](https://example.test/path)"),
    ("****1234", r"\*\*\*\*1234"),
    ("`0001`_code_", r"\`0001\`\_code\_"),
    ("first\r\nsecond\rthird\nfourth", "first ↵ second ↵ third ↵ fourth"),
])
def test_business_markdown_displays_source_markup_literally_without_html_or_masking(tmp_path, value, display) -> None:
    bundle, _ = _business_bundle()
    row = bundle.datasets[0].rows[0]
    row["normalized"]["summary"] = value
    row["raw"]["摘要"] = value
    payload = bundle.json_payload()
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=bundle.document["id"])
    markdown = paths["enhanced_reading"].read_text(encoding="utf-8")
    assert display in markdown.split("## 提取说明")[0]
    assert_business_markdown_values(payload, markdown)
    assert validate_community_artifacts(paths["community"]) == []
