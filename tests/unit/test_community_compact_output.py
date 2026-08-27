"""Source-aware compact exports must not change business values or evidence."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult, TextBlock
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import CommunityDataset, project_community_bundle
from docmirror.plugins._base.projector import load_projection_policy
from docmirror.server.edition_outputs import _write_community_bundle_files


def _column(key: str, **extra) -> dict:
    return {
        "key": key,
        "label": key,
        "type": "string",
        "nullable": True,
        "raw_available": False,
        "evidence_available": False,
        **extra,
    }


def _record(**normalized) -> dict:
    return {
        "record_id": "records:r000001",
        "normalized": {"date": "2026-01-02", **normalized},
        "canonical_raw": {"date": "20260102"},
        "raw": {"交易日期": "20260102"},
        "source": {"page_range": [1, 1], "evidence_ids": ["cell:date"]},
    }


def _dataset(rows: list[dict], *keys: str) -> CommunityDataset:
    return CommunityDataset(
        public={"id": "ds_transactions", "columns": [_column(key) for key in ("date", *keys)]},
        rows=rows,
    )


def test_compact_omits_only_unsupported_placeholders_and_keeps_source_layers() -> None:
    row = _record(amount=0, enabled=False, summary="", unknown=None, empty_list=[], empty_object={})
    row["canonical_raw"]["source_blank"] = None
    row["source"]["field_sources"] = {"unreadable": {"status": "unreadable"}}
    dataset = _dataset(
        [row],
        "amount",
        "enabled",
        "summary",
        "unknown",
        "empty_list",
        "empty_object",
        "source_blank",
        "unreadable",
        "schema_only",
    )
    before = copy.deepcopy(dataset)
    dense = dataset.to_payload()
    compact = dataset.to_payload(compact=True)

    assert compact["omitted_normalized_fields"] == ["summary", "schema_only"]
    normalized = compact["rows"][0]["normalized"]
    assert "summary" not in normalized and "schema_only" not in normalized
    assert normalized["amount"] == 0
    assert normalized["enabled"] is False
    assert normalized["unknown"] is None
    assert normalized["source_blank"] is None
    assert normalized["unreadable"] is None
    assert normalized["empty_list"] == "[]"
    assert normalized["empty_object"] == "{}"
    for key in ("record_id", "canonical_raw", "raw", "source"):
        assert compact["rows"][0][key] == dense["rows"][0][key]
    assert compact["columns"] == dense["columns"]
    assert compact["completeness"] == dense["completeness"]
    assert dataset == before
    assert dataset.to_payload() == dense


@pytest.mark.parametrize("header", ["对方户名", "对 方 户 名：", "对方户名\nCounterparty Name", "收（付）方名称"])
def test_printed_blank_columns_survive_alias_and_bilingual_matching(header: str) -> None:
    row = _record(counter_party="")
    row["raw"][header] = ""
    dataset = _dataset([row], "counter_party", "schema_only")
    compact = dataset.to_payload(
        compact=True,
        source_aliases={"counter_party": ["对方户名", "Counterparty Name", "收(付)方名称"]},
    )
    assert compact["rows"][0]["normalized"]["counter_party"] == ""
    assert compact["omitted_normalized_fields"] == ["schema_only"]


def test_unknown_blank_source_header_prevents_unjustified_omission() -> None:
    row = _record(summary="")
    row["raw"]["尚未映射的原始列"] = ""
    dataset = _dataset([row], "summary", "schema_only")
    assert dataset.to_payload(compact=True) == dataset.to_payload()


def test_field_present_on_later_row_keeps_column_and_earlier_placeholders() -> None:
    first = _record(counter_party="")
    second = _record(counter_party="示例商户")
    second["record_id"] = "records:r000002"
    second["raw"]["对方户名"] = "示例商户"
    dataset = _dataset([first, second], "counter_party", "schema_only")
    compact = dataset.to_payload(compact=True)
    assert [row["normalized"]["counter_party"] for row in compact["rows"]] == ["", "示例商户"]
    assert compact["omitted_normalized_fields"] == ["schema_only"]


def test_required_fields_foreign_keys_and_derived_uncertainty_are_not_hidden() -> None:
    row = _record(derived="")
    row["source"]["field_sources"] = {"derived": {"normalized_only": True, "status": "unknown"}}
    dataset = _dataset([row], "required", "nonnullable", "header_id", "derived", "schema_only")
    dataset.public["columns"][1]["required"] = True
    dataset.public["columns"][2]["nullable"] = False
    dataset.public["foreign_keys"] = [{"columns": ["header_id"]}]
    compact = dataset.to_payload(compact=True)
    assert compact["omitted_normalized_fields"] == ["schema_only"]


def test_no_source_plane_keeps_explicit_empty_values() -> None:
    row = {"normalized": {"name": "", "unknown": None}}
    dataset = _dataset([row], "name", "unknown", "schema_only")
    compact = dataset.to_payload(compact=True)
    assert compact["rows"][0]["normalized"] == {"name": "", "unknown": None}


def test_empty_datasets_do_not_claim_source_absence() -> None:
    dataset = _dataset([], "schema_only")
    assert dataset.to_payload(compact=True) == dataset.to_payload()


def _bundle(*, compact: bool):
    row = _record(amount=0, direction="income", summary="", counter_party="")
    row["raw"]["对方户名"] = ""
    policy = {
        "omit_absent_fields": True,
        "minify_json": True,
        "source_aliases": {"transactions": {"counter_party": ["对方户名"]}},
    }
    source = ParseResult(
        entities=DocumentEntities(document_type="bank_statement"),
        pages=[PageContent(page_number=1, texts=[TextBlock(content="原始内容保持不变")])],
    )
    sealed = seal_parse_result(source)
    bundle = project_community_bundle(
        sealed,
        projection_policy=load_projection_policy("docmirror.plugins.bank_statement"),
        projection_data={
            "projector_id": "compact-export-test",
            "document_type": "bank_statement",
            "datasets": {"records": [row]},
            "domain_facts": {
                "data_dictionary": {
                    "record_columns": {
                        key: {"label": label}
                        for key, label in (
                            ("date", "日期"),
                            ("amount", "金额"),
                            ("direction", "方向"),
                            ("counter_party", "对方户名"),
                            ("summary", "不存在的摘要"),
                            ("schema_only", "不存在的字段"),
                        )
                    }
                }
            },
            "semantic": {"compact_output": policy} if compact else {},
        },
    )
    return bundle, sealed


def test_compact_bundle_keeps_csv_audit_source_markdown_and_schema_contracts() -> None:
    bundle, sealed = _bundle(compact=True)
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    dense = replace(bundle, domain=dense_domain)
    semantic = bundle.semantic_payload()
    dense_semantic = dense.semantic_payload()
    payload = bundle.json_payload(semantic)

    assert validate_projection_payload("community_semantic", semantic).valid
    assert validate_projection_payload("community", payload).valid
    assert semantic["datasets"] == payload["datasets"]
    assert semantic["structure"] == dense_semantic["structure"]
    assert semantic["bindings"] == dense_semantic["bindings"]
    assert semantic["warnings"] == dense_semantic["warnings"]
    assert bundle.render_dataset_csvs(semantic) == dense.render_dataset_csvs(dense_semantic)
    assert bundle.render_audit_csv(semantic) == dense.render_audit_csv(dense_semantic)
    assert bundle.render_markdown() == dense.render_markdown()
    assert bundle.conservation_issues(payload=payload, dataset_csvs=bundle.render_dataset_csvs(semantic)) == []
    enhanced = bundle.render_enhanced_markdown(semantic)
    assert "对方户名" in enhanced
    assert "不存在的摘要" not in enhanced
    assert "不存在的字段" not in enhanced
    assert "不存在的摘要" in dense.render_enhanced_markdown(dense_semantic)
    assert sealed.verify_integrity()


@pytest.mark.parametrize("compact", [False, True])
def test_writer_minifies_only_opted_in_community_json(tmp_path, compact: bool) -> None:
    bundle, sealed = _bundle(compact=compact)
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id="doc_test")
    written = paths["community"].read_text(encoding="utf-8")
    payload = json.loads(written)
    assert payload == bundle.json_payload()
    assert ("\n  " not in written) is compact
    assert payload["datasets"][0]["rows"][0]["normalized"]["amount"] == 0
    assert paths["content"].read_text(encoding="utf-8") == bundle.render_markdown()
    assert validate_projection_payload("community", payload).valid
    assert sealed.verify_integrity()


def test_corpus_export_validator_uses_production_files_and_dense_baseline(tmp_path) -> None:
    from scripts.validate.bank_compact_exports import validate_and_write_bank_exports

    bundle, _sealed = _bundle(compact=True)
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    baseline = replace(bundle, domain=dense_domain).json_payload()
    payload, report = validate_and_write_bank_exports(bundle, tmp_path, baseline=baseline)
    assert report["status"] == "pass"
    assert report["baseline_checked"] is True
    assert report["compact_json_bytes"] < report["dense_json_bytes"]
    assert report["compact_enhanced_markdown_bytes"] < report["dense_enhanced_markdown_bytes"]
    assert payload["datasets"][0]["rows"][0]["normalized"]["amount"] == 0


@pytest.mark.parametrize("corruption", ["amount", "source", "row_count", "false_type"])
def test_compact_validator_detects_business_evidence_and_type_changes(corruption: str) -> None:
    from scripts.validate.bank_compact_exports import assert_value_preserving_compaction

    bundle, _sealed = _bundle(compact=False)
    dense = bundle.json_payload()
    corrupted = copy.deepcopy(dense)
    dataset = corrupted["datasets"][0]
    if corruption == "amount":
        dataset["rows"][0]["normalized"]["amount"] = 999
    elif corruption == "source":
        dataset["rows"][0]["source"] = {}
    elif corruption == "row_count":
        dataset["rows"] = []
    else:
        dataset["rows"][0]["normalized"]["amount"] = False
    with pytest.raises(AssertionError, match="changed data"):
        assert_value_preserving_compaction(dense, corrupted)


def test_compact_json_replay_does_not_reintroduce_absent_placeholders(tmp_path) -> None:
    from docmirror.server.output_builder import materialize_community_bundle

    bundle, _sealed = _bundle(compact=True)
    payload = bundle.json_payload()
    restored = materialize_community_bundle(payload, bundle.result)

    assert restored.json_payload() == payload
    paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=payload["document"]["id"])
    written = paths["community"].read_text(encoding="utf-8")
    assert json.loads(written) == payload
    assert "\n  " not in written


def test_compact_replay_preserves_unmapped_source_blanks_without_inventing_canonical_raw() -> None:
    from docmirror.server.output_builder import materialize_community_bundle

    bundle, _sealed = _bundle(compact=True)
    # The source prints this blank field, but extraction did not assign a
    # canonical value. Schema expansion supplies a normalized null only.
    bundle.datasets[0].rows[0]["normalized"].pop("counter_party")
    payload = bundle.json_payload()
    row = payload["datasets"][0]["rows"][0]
    assert row["normalized"]["counter_party"] is None
    assert "counter_party" not in row["canonical_raw"]
    restored = materialize_community_bundle(payload, bundle.result)
    assert restored.json_payload() == payload


@pytest.mark.parametrize("value", ["新增业务摘要", 0, False])
def test_stale_omission_metadata_cannot_hide_a_populated_business_field(value) -> None:
    from docmirror.output.community_bundle import render_community_reading_markdown
    from docmirror.server.output_builder import materialize_community_bundle

    bundle, _sealed = _bundle(compact=True)
    payload = bundle.json_payload()
    payload["datasets"][0]["rows"][0]["normalized"]["summary"] = value
    # The existing reading plan omitted this field too. A caller explicitly
    # requesting it must not have a real value hidden by stale metadata.
    payload["reading"]["tables"][0]["column_keys"].append("summary")
    assert "不存在的摘要" in render_community_reading_markdown(payload)
    restored = materialize_community_bundle(payload, bundle.result)
    restored_payload = restored.json_payload()
    dataset = restored_payload["datasets"][0]
    assert "summary" not in dataset.get("omitted_normalized_fields", [])
    assert dataset["rows"][0]["normalized"]["summary"] == value
    assert type(dataset["rows"][0]["normalized"]["summary"]) is type(value)
    assert "summary" in restored_payload["reading"]["tables"][0]["column_keys"]


@pytest.mark.parametrize("declaration", [["summary", "summary"], [123]])
def test_schema_rejects_malformed_omission_declarations(declaration) -> None:
    bundle, _sealed = _bundle(compact=True)
    payload = bundle.json_payload()
    payload["datasets"][0]["omitted_normalized_fields"] = declaration
    assert not validate_projection_payload("community", payload).valid


def test_saved_corpus_replay_checks_values_without_running_extraction(tmp_path, monkeypatch) -> None:
    from docmirror.plugins.bank_statement.community_plugin import plugin
    from scripts.validate.bank_compact_exports import replay_export_report, validate_and_write_bank_exports

    bundle, _sealed = _bundle(compact=True)
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    baseline = replace(bundle, domain=dense_domain).json_payload()
    _payload, validation = validate_and_write_bank_exports(bundle, tmp_path / "exports", baseline=baseline)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "filename": "synthetic.pdf",
                        "source_sha256": "synthetic",
                        "export_validation": validation,
                        "community": validation["artifacts"]["community"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def unexpected_extraction(*_args, **_kwargs):
        raise AssertionError("saved-output replay must not invoke extraction")

    monkeypatch.setattr(plugin, "derive", unexpected_extraction)
    result = replay_export_report(report_path)
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["extraction_executed"] is False


def _normalized_bundle():
    bundle, sealed = _bundle(compact=True)
    bundle.domain["extensions"]["compact_output"]["normalized_only"] = True
    return bundle, sealed


def test_normalized_only_export_retains_internal_evidence_and_all_business_values(tmp_path) -> None:
    from scripts.validate.bank_compact_exports import validate_and_write_bank_exports

    bundle, sealed = _normalized_bundle()
    row = bundle.datasets[0].rows[0]
    row["raw"].update({"核心流水号": "0000123400567890123456", "交易网点": "测试支行", "交易介质编号": "0007"})
    before = copy.deepcopy(bundle.datasets)
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    baseline = replace(bundle, domain=dense_domain).json_payload()
    payload, report = validate_and_write_bank_exports(bundle, tmp_path, baseline=baseline)
    public_row = payload["datasets"][0]["rows"][0]
    internal_row = bundle.semantic_payload()["datasets"][0]["rows"][0]
    extras = {item["name"]: item["value"] for item in public_row["normalized"]["additional_fields"]}
    assert extras["核心流水号"] == "0000123400567890123456"
    assert extras["交易网点"] == "测试支行"
    assert extras["交易介质编号"] == "0007"
    assert "raw" not in public_row and "canonical_raw" not in public_row
    assert internal_row["raw"] == baseline["datasets"][0]["rows"][0]["raw"]
    assert internal_row["canonical_raw"] == baseline["datasets"][0]["rows"][0]["canonical_raw"]
    assert public_row["normalized"]["amount"] == 0
    assert report["raw_only_business_fields_accounted_for"] is True
    assert report["existing_csv_fields_unchanged"] is True
    assert "核心流水号" in bundle.render_enhanced_markdown()
    assert bundle.datasets == before
    assert sealed.verify_integrity()


@pytest.mark.parametrize(
    ("source", "normalized", "descriptor", "expected"),
    [
        (False, 0, {"type": "integer"}, False),
        (0, False, {"type": "integer"}, False),
        ("000123", "123", {"type": "string", "format": "long_id"}, False),
        ("001", "1", {"type": "string"}, False),
        ("1,234.00", "1234.0", {"type": "money"}, True),
        ("12,34.00", "1234.0", {"type": "money"}, False),
        ("-25.00", "25.0", {"type": "money"}, False),
        ("NaN", "0", {"type": "money"}, False),
        ("20260102", "2026-01-02", {"type": "date"}, True),
        ("20260230", "2026-02-28", {"type": "date"}, False),
        ("2026/01/02 12:03:04", "2026-01-02T12:03:04", {"type": "datetime"}, True),
        ("AB C", "A BC", {"type": "string"}, False),
        ("A\nB", "A B", {"type": "string"}, True),
        (["20260102", "20260102"], "2026-01-02", {"type": "date"}, True),
        (["20260102", "20260202"], "2026-01-02", {"type": "date"}, False),
        ("[1]", 1, {"type": "integer"}, False),
        ({"value": 7}, 7, {"type": "integer"}, False),
        (0, "0", {"type": "money"}, True),
    ],
)
def test_normalized_source_coverage_does_not_guess_business_semantics(source, normalized, descriptor, expected):
    from docmirror.output.normalized_records import value_is_represented

    assert value_is_represented(source, normalized, descriptor) is expected


@pytest.mark.parametrize("value", [0, False, None, "", [], {}, "=SUM(A1:A9)", "00000012345678901234567"])
def test_unknown_source_fields_are_preserved_even_when_an_unrelated_field_matches(value) -> None:
    from docmirror.output.normalized_records import additional_business_fields

    row = _record(summary=value)
    row["raw"]["未定义银行业务列"] = value
    extras = additional_business_fields(row, [_column("date"), _column("summary")], {})
    assert any(item["name"] == "未定义银行业务列" and item["value"] == value for item in extras)
    actual = next(item["value"] for item in extras if item["name"] == "未定义银行业务列")
    assert type(actual) is type(value)


def test_compound_and_canonical_only_business_values_survive_without_overwriting_standard_fields() -> None:
    from docmirror.output.normalized_records import additional_business_fields

    row = _record(counter_party="示例公司", amount="25.0", posting_date="")
    row["raw"] = {"交易对手信息": "示例公司 / 000123 / 测试支行", "交易金额": "-25.00"}
    row["canonical_raw"] = {"amount": "-25.00", "posting_date": "2023-08-15", "counter_party": "示例公司"}
    columns = [
        _column("counter_party", label="交易对手信息"),
        _column("amount", label="交易金额", type="money"),
        _column("posting_date", type="date"),
    ]
    before = copy.deepcopy(row)
    assert additional_business_fields(row, columns, {}) == [
        {"name": "交易对手信息", "value": "示例公司 / 000123 / 测试支行"},
        {"name": "交易金额", "value": "-25.00", "field": "amount"},
        {"name": "posting_date", "field": "posting_date", "value": "2023-08-15"},
    ]
    assert row == before


def test_bilingual_known_heading_does_not_become_an_unmapped_business_field() -> None:
    from docmirror.output.normalized_records import additional_business_fields

    row = _record(amount="123.0")
    row["raw"] = {"交易金额\nTransaction\nAmount": "123.00"}
    row["canonical_raw"] = {"amount": "123.00"}
    assert additional_business_fields(row, [_column("amount", label="交易金额", type="money")], {}) == []


def test_v4_replay_keeps_native_supplemental_fields_sparse_rows_and_no_fabricated_raw(tmp_path) -> None:
    import csv
    import io

    from docmirror.server.output_builder import materialize_community_bundle

    bundle, _ = _normalized_bundle()
    first = bundle.datasets[0].rows[0]
    first["raw"]["新业务字段"] = {"codes": ["0001", "0002"], "active": False}
    second = copy.deepcopy(first)
    second["record_id"] = "records:r000002"
    second["raw"].pop("新业务字段")
    bundle.datasets[0].rows.append(second)
    payload = bundle.json_payload()
    restored = materialize_community_bundle(payload, ParseResult())
    assert restored.json_payload() == payload
    replay_internal = restored.semantic_payload()
    assert all(row["raw"] == row["canonical_raw"] == {} for row in replay_internal["datasets"][0]["rows"])
    assert all(row["raw"] == "" for row in csv.DictReader(io.StringIO(restored.render_audit_csv())))
    paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=payload["document"]["id"])
    assert json.loads(paths["community"].read_text(encoding="utf-8")) == payload
    dataset_csv = next(iter(restored.render_dataset_csvs().values()))
    csv_rows = list(csv.DictReader(io.StringIO(dataset_csv.lstrip("\ufeff"))))
    assert json.loads(csv_rows[0]["additional_fields"]) == payload["datasets"][0]["rows"][0]["normalized"]["additional_fields"]


@pytest.mark.parametrize("corruption", ["raw", "canonical_raw", "additional_fields", "extra_property", "missing_name"])
def test_v4_schema_rejects_source_pools_and_malformed_supplemental_fields(corruption) -> None:
    bundle, _ = _normalized_bundle()
    payload = bundle.json_payload()
    row = payload["datasets"][0]["rows"][0]
    if corruption in {"raw", "canonical_raw"}:
        row[corruption] = {}
    elif corruption == "additional_fields":
        row["normalized"]["additional_fields"] = "[]"
    elif corruption == "extra_property":
        row["normalized"]["additional_fields"] = [{"name": "未知字段", "value": 0, "invented": True}]
    else:
        row["normalized"]["additional_fields"] = [{"value": 0}]
    assert not validate_projection_payload("community", payload).valid


def test_legacy_schema_still_requires_both_source_value_pools() -> None:
    bundle, _ = _bundle(compact=False)
    payload = bundle.json_payload()
    assert payload["schema"]["version"] == "3.0.0"
    for key in ("raw", "canonical_raw"):
        broken = copy.deepcopy(payload)
        broken["datasets"][0]["rows"][0].pop(key)
        assert not validate_projection_payload("community", broken).valid


def test_normalized_scalar_items_preserve_non_equivalent_source_values_without_raw() -> None:
    from docmirror.output.normalized_records import strip_source_value_pools

    payload = {"schema": {}, "sections": [{"items": [
        {"key": "amount", "value": "25.0", "raw": "25.00", "type": "money"},
        {"key": "period", "value": "2026-01", "raw": "2026-01-02 ~ 2026-01-31", "type": "string"},
    ], "groups": [{"items": [{"key": "reference", "value": "123", "raw": "00123", "type": "string"}]}]}]}
    strip_source_value_pools(payload)
    section = payload["sections"][0]
    assert "additional_values" not in section["items"][0]
    assert section["items"][1]["additional_values"] == ["2026-01-02 ~ 2026-01-31"]
    assert section["groups"][0]["items"][0]["additional_values"] == ["00123"]
    assert all("raw" not in item for item in section["items"])


@pytest.mark.parametrize("corruption", ["remove_extra", "invent_extra", "amount", "zero_type", "source"])
def test_normalized_export_audit_detects_loss_invention_and_existing_field_changes(corruption) -> None:
    from scripts.validate.bank_compact_exports import assert_value_preserving_compaction

    bundle, _ = _normalized_bundle()
    bundle.datasets[0].rows[0]["raw"]["新业务字段"] = "0007"
    domain = copy.deepcopy(bundle.domain)
    domain["extensions"].pop("compact_output")
    dense = replace(bundle, domain=domain).json_payload()
    payload = bundle.json_payload()
    row = payload["datasets"][0]["rows"][0]
    if corruption == "remove_extra":
        row["normalized"]["additional_fields"] = []
    elif corruption == "invent_extra":
        row["normalized"]["additional_fields"].append({"name": "虚构字段", "value": "虚构值"})
    elif corruption == "amount":
        row["normalized"]["amount"] = "500"
    elif corruption == "zero_type":
        row["normalized"]["amount"] = False
    else:
        row["source"] = {}
    with pytest.raises(AssertionError):
        assert_value_preserving_compaction(dense, payload)


def test_evidence_binding_requires_identical_values_types_sources_and_record_identity() -> None:
    from scripts.validate.bank_compact_exports import bind_internal_evidence

    bundle, _ = _normalized_bundle()
    payload, evidence = bundle.json_payload(), bundle.semantic_payload()
    bound = bind_internal_evidence(payload, evidence)
    assert bound["datasets"][0]["rows"][0]["raw"] == evidence["datasets"][0]["rows"][0]["raw"]
    for key, value in (("record_id", "other"), ("source", {}), ("normalized", {"amount": False})):
        broken = copy.deepcopy(payload)
        broken["datasets"][0]["rows"][0][key] = value
        with pytest.raises(AssertionError, match="internal evidence record"):
            bind_internal_evidence(broken, evidence)


def test_unmasked_bank_identifiers_survive_supplemental_fields_json_and_replay(tmp_path) -> None:
    from docmirror.output.community_bundle import render_community_reading_markdown
    from docmirror.server.output_builder import materialize_community_bundle

    bundle, _ = _normalized_bundle()
    bundle.domain["extensions"]["enhanced_markdown"] = {"privacy_mode": "full"}
    dataset = bundle.datasets[0]
    dataset.public["columns"].append(_column("account_number", label="账号", sensitive=True, display="masked"))
    source_account = "0001 2345 6789 9876"
    dataset.rows[0]["normalized"]["account_number"] = source_account.replace(" ", "")
    dataset.rows[0]["raw"]["账号"] = source_account
    dataset.rows[0]["canonical_raw"]["account_number"] = source_account
    payload = bundle.json_payload()
    extras = payload["datasets"][0]["rows"][0]["normalized"]["additional_fields"]
    account = next(item for item in extras if item["name"] == "账号")
    assert account == {"name": "账号", "value": source_account, "field": "account_number"}
    assert validate_projection_payload("community", payload).valid
    assert payload["reading"]["privacy_mode"] == "full"
    restored = materialize_community_bundle(payload, ParseResult())
    assert restored.json_payload() == payload
    paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=payload["document"]["id"])
    for markdown in (
        bundle.render_enhanced_markdown(),
        render_community_reading_markdown(payload),
        restored.render_enhanced_markdown(),
        paths["enhanced_reading"].read_text(encoding="utf-8"),
    ):
        assert source_account in markdown
        assert source_account.replace(" ", "") in markdown


def test_full_markdown_preserves_source_redactions_without_inventing_digits() -> None:
    bundle, _ = _normalized_bundle()
    bundle.domain["extensions"]["enhanced_markdown"] = {"privacy_mode": "full"}
    dataset = bundle.datasets[0]
    dataset.public["columns"].append(_column("account_number", label="账号", sensitive=True, display="masked"))
    dataset.rows[0]["normalized"]["account_number"] = "6210****4321"
    dataset.rows[0]["raw"]["账号"] = "6210****4321"
    assert "6210****4321" in bundle.render_enhanced_markdown()


def test_default_markdown_privacy_of_other_outputs_is_unchanged() -> None:
    from docmirror.output.community_bundle import render_community_reading_markdown

    bundle, _ = _bundle(compact=False)
    dataset = bundle.datasets[0]
    dataset.public["columns"].append(_column("account_number", sensitive=True, display="masked"))
    dataset.rows[0]["normalized"]["account_number"] = "0001234567899876"
    assert "privacy_mode" not in bundle.json_payload()["reading"]
    assert "0001234567899876" not in bundle.render_enhanced_markdown()
    assert "0001234567899876" not in render_community_reading_markdown(bundle.json_payload())


@pytest.mark.parametrize("mode", ["full", "masked", "invalid"])
def test_public_reading_privacy_mode_contract(mode) -> None:
    bundle, _ = _normalized_bundle()
    payload = bundle.json_payload()
    payload["reading"]["privacy_mode"] = mode
    assert validate_projection_payload("community", payload).valid is (mode != "invalid")


@pytest.mark.parametrize("corruption", [None, "checksum", "normalized"])
@pytest.mark.parametrize("business_only", [False, True])
def test_unmasked_artifact_refresh_preserves_facts_without_extraction(tmp_path, monkeypatch, corruption, business_only) -> None:
    import hashlib

    from docmirror.plugins.bank_statement.community_plugin import plugin
    from docmirror.server.artifact_writer import ArtifactWriter
    from scripts.validate.bank_compact_exports import (
        refresh_bank_markdown_report,
        replay_export_report,
        validate_and_write_bank_exports,
    )

    bundle, _ = _normalized_bundle()
    dataset = bundle.datasets[0]
    dataset.public["columns"].append(_column("account_number", label="账号", sensitive=True, display="masked"))
    dataset.rows[0]["normalized"]["account_number"] = "0001234567899876"
    dataset.rows[0]["raw"]["账号"] = "0001234567899876"
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output")
    baseline = replace(bundle, domain=dense_domain).json_payload()
    source_dir = tmp_path / "original"
    payload, validation = validate_and_write_bank_exports(bundle, source_dir / "exports", baseline=baseline)
    writer = ArtifactWriter(source_dir)
    cache = writer.write_json("projection.community.json", payload)
    evidence = bundle.semantic_payload()
    if corruption == "normalized":
        evidence["datasets"][0]["rows"][0]["normalized"]["amount"] = 999
    evidence_path = writer.write_json("projection.community.evidence.json", evidence)
    writer.write_json("projection.meta.json", {
        "evidence_sha256": "invalid" if corruption == "checksum" else hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    })
    source_report = writer.write_json("report.json", {"results": [{
        "filename": "synthetic.pdf", "source_sha256": "0" * 64,
        "community": str(cache), "export_validation": validation,
        "audit_status": "pass", "error_count": 0,
    }]})
    original_bytes = {path: path.read_bytes() for path in source_dir.rglob("*") if path.is_file()}

    def reject_extraction(*_args, **_kwargs):
        raise AssertionError("presentation refresh must not execute extraction")

    monkeypatch.setattr(plugin, "project_bundle", reject_extraction)
    monkeypatch.setattr(plugin, "derive", reject_extraction)
    refreshed = refresh_bank_markdown_report(source_report, tmp_path / "refreshed", business_only=business_only)
    assert refreshed["extraction_executed"] is False
    assert original_bytes == {path: path.read_bytes() for path in original_bytes}
    if corruption:
        assert refreshed["failed"] == 1
        assert refreshed["passed"] == 0
        expected = "checksum" if corruption == "checksum" else "normalized"
        assert expected in refreshed["results"][0]["error"]
        return
    assert refreshed["passed"] == 1
    assert refreshed["failed"] == 0
    result = refreshed["results"][0]
    from pathlib import Path
    assert "0001234567899876" in Path(result["export_validation"]["artifacts"]["enhanced_reading"]).read_text(encoding="utf-8")
    refreshed_report = ArtifactWriter(tmp_path).write_json("refreshed_report.json", refreshed)
    replay = replay_export_report(refreshed_report)
    assert replay["failed"] == 0
    assert replay["results"][0]["unmasked_markdown_preserved"] is True
    if business_only:
        assert replay["results"][0]["business_view_checked"] is True
    with pytest.raises(FileExistsError):
        refresh_bank_markdown_report(source_report, tmp_path / "refreshed")
