# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for bank statement header/context Community projection."""

from __future__ import annotations

import json
from types import SimpleNamespace

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins._base.projector import load_projection_policy
from docmirror.plugins.bank_statement import community_plugin as community_module
from docmirror.plugins.bank_statement.canonical import StyleMeta
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.extract_pipeline import BankExtractResult
from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.statement_context import attach_statement_context
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence


def _detail(
    raw_name: str,
    raw_value: str,
    normalized_value: str | None = None,
    *,
    page: int = 1,
) -> dict:
    return {
        "raw_name": raw_name,
        "raw_value": raw_value,
        "normalized_value": normalized_value if normalized_value is not None else raw_value,
        "data_type": "string",
        "source": "test.header",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "page": page,
                "page_range": [page, page],
                "bbox": [10.0, 10.0, 100.0, 20.0],
            }
        ],
        "evidence_ids": [f"header:p{page}:{raw_name}"],
    }


def _transaction_record(*, own_account: str = "", currency: str = "") -> dict:
    return {
        "record_id": "records:r000001",
        "raw": {
            "交易日期": "2023-02-24",
            "收入": "100.00",
            "余额": "110.00",
        },
        "canonical_raw": {
            "date": "2023-02-24",
            "amount": "100.00",
            "balance": "110.00",
            "own_account": own_account,
            "currency": currency,
        },
        "normalized": {
            "date": "2023-02-24",
            "amount": 100.0,
            "balance": 110.0,
            "direction": "income",
            "own_account": own_account,
            "currency": currency,
        },
        "source": {
            "source_page": 1,
            "page_range": [1, 1],
            "source_refs": [{"source": "native_pdf_words", "page": 1}],
        },
    }


def _parse_result() -> ParseResult:
    return ParseResult(
        pages=[PageContent(page_number=1, source_page_number=1)],
        entities=DocumentEntities(document_type="bank_statement"),
    )


def _synthetic_extract_result() -> BankExtractResult:
    return BankExtractResult(
        ctx=SimpleNamespace(full_text="", reconstruction=ReconstructionMeta(source="none")),
        detection=SimpleNamespace(),
        records=[_transaction_record()],
        identity_fields={
            "statement_title": _detail("标题", "活期账户明细查询"),
            "account_holder": _detail("户名", "测试工具有限公司"),
            "account_number": _detail("账号", "3211020801201000170968"),
            "currency": _detail("币种", "人民币", "CNY"),
            "query_period": _detail("交易时段", "2023-02-23 至 2023-05-22"),
            "print_date": _detail("打印日期", "2023/08/24", "2023-08-24"),
        },
        style_meta=StyleMeta(
            style_id="split_debit_credit",
            style_confidence=1.0,
            expected_primary_rows=1,
            extracted_rows=1,
            coverage_ratio=1.0,
            canonical_expected=1,
            canonical_extracted=1,
            canonical_ratio=1.0,
        ),
        warnings=[],
        parsed_rows=1,
        canonical_rows=1,
        emitted_rows=1,
        extraction_route=BankExtractionRoute.DIGITAL,
    )


def _project_synthetic_result(monkeypatch, parse_result: ParseResult, result: BankExtractResult):
    monkeypatch.setattr(community_module, "run_bank_statement_extract", lambda *_args: result)
    monkeypatch.setattr(
        community_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence.empty(),
    )
    projection = BankStatementCommunityPlugin().derive(parse_result, "")
    bundle = project_community_bundle(
        seal_parse_result(parse_result),
        projection_data=projection.model_dump(mode="python"),
        projection_policy=load_projection_policy("docmirror.plugins.bank_statement"),
    )
    return projection, bundle, bundle.json_payload()


def test_community_json_orders_statement_header_before_transactions(monkeypatch) -> None:
    parse_result = _parse_result()
    monkeypatch.setattr(
        community_module,
        "run_bank_statement_extract",
        lambda *_args: _synthetic_extract_result(),
    )
    monkeypatch.setattr(
        community_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence.empty(),
    )

    projection = BankStatementCommunityPlugin().derive(parse_result, "")
    bundle = project_community_bundle(
        seal_parse_result(parse_result),
        projection_data=projection.model_dump(mode="python"),
        projection_policy=load_projection_policy("docmirror.plugins.bank_statement"),
    )
    payload = bundle.json_payload()

    assert [dataset["name"] for dataset in payload["datasets"]] == [
        "statement_header",
        "transactions",
    ]
    assert payload["sections"][0]["dataset_refs"] == [
        "ds_statement_header",
        "ds_transactions",
    ]
    assert [table["dataset_id"] for table in payload["reading"]["tables"]] == [
        "ds_statement_header",
        "ds_transactions",
    ]
    header = payload["datasets"][0]["rows"][0]
    expected_header_values = {
        "account_holder": "测试工具有限公司",
        "account_number": "3211020801201000170968",
        "currency": "CNY",
        "period_end": "2023-05-22",
        "period_start": "2023-02-23",
        "print_date": "2023-08-24",
        "query_period": "2023-02-23 ~ 2023-05-22",
        "statement_title": "活期账户明细查询",
    }
    assert {
        key: header["normalized"][key] for key in expected_header_values
    } == expected_header_values
    internal_header = bundle.semantic_payload()["datasets"][0]["rows"][0]
    assert "raw" not in header and "canonical_raw" not in header
    assert internal_header["raw"]["币种"] == "人民币"
    assert internal_header["canonical_raw"]["currency"] == "人民币"
    assert internal_header["source"]["field_sources"]["account_number"]["evidence_ids"] == [
        "header:p1:账号"
    ]
    assert "style_id" not in header["normalized"]
    assert "extraction_route" not in header["normalized"]

    transaction = payload["datasets"][1]["rows"][0]["normalized"]
    assert payload["datasets"][1]["foreign_keys"] == [
        {
            "columns": ["extraction.statement_header_id"],
            "reference_dataset": "statement_header",
            "reference_columns": ["extraction.record_id"],
        }
    ]
    assert "statement_header_id" not in transaction
    assert payload["datasets"][1]["rows"][0]["extraction"]["statement_header_id"] == header["extraction"]["record_id"]
    assert transaction["own_account"] == "3211020801201000170968"
    assert transaction["account_holder"] == "测试工具有限公司"
    assert transaction["currency"] == "CNY"
    assert transaction["statement_title"] == "活期账户明细查询"
    assert transaction["period_start"] == "2023-02-23"
    assert transaction["period_end"] == "2023-05-22"
    assert transaction["print_date"] == "2023-08-24"
    assert validate_projection_payload("community", payload).valid
    assert bundle.conservation_issues(payload=payload) == []


def test_digital_bank_unmasks_all_identifier_fields_and_scanned_default_is_unchanged(monkeypatch) -> None:
    from docmirror.output.community_bundle import render_community_reading_markdown
    from docmirror.server.output_builder import materialize_community_bundle

    result = _synthetic_extract_result()
    identifiers = {
        "account_number": ("账号", "3211020801201000170968"),
        "card_number": ("卡号", "6222123400005678901"),
        "internal_account": ("内部账号", "000123459876"),
        "customer_number": ("客户号", "000123456789"),
        "id_number": ("证件号码", "110101199001021234"),
    }
    for key, (label, value) in identifiers.items():
        result.identity_fields[key] = _detail(label, value)
    projection, bundle, payload = _project_synthetic_result(monkeypatch, _parse_result(), result)
    assert projection.semantic["enhanced_markdown"]["privacy_mode"] == "full"
    assert payload["reading"]["privacy_mode"] == "full"
    replay = materialize_community_bundle(payload, ParseResult())
    assert replay.json_payload() == payload
    def reject_application_mask(_value):
        raise AssertionError("digital-bank Markdown must not invoke masking")

    monkeypatch.setattr("docmirror.output.community_bundle._masked_display", reject_application_mask)
    for markdown in (
        bundle.render_enhanced_markdown(),
        render_community_reading_markdown(payload),
        replay.render_enhanced_markdown(),
    ):
        for _label, value in identifiers.values():
            assert value in markdown
    result.extraction_route = BankExtractionRoute.SCANNED
    scanned, _, scanned_payload = _project_synthetic_result(monkeypatch, _parse_result(), result)
    assert "enhanced_markdown" not in scanned.semantic
    assert "privacy_mode" not in scanned_payload["reading"]
    assert scanned_payload["schema"]["version"] == "3.0.0"


def test_normalized_only_source_period_envelope_preserves_exact_header_components(monkeypatch) -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(page_number=1, source_page_number=1),
            PageContent(page_number=2, source_page_number=2),
        ],
        entities=DocumentEntities(document_type="bank_statement"),
    )
    result = _synthetic_extract_result()
    result.identity_fields["query_period"] = {
        "raw_name": "source period components",
        "normalized_value": "2023-06-01 至 2023-07-31",
        "data_type": "string",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {"source": "canonical_evidence_atoms", "page_id": "page:0001"},
            {"source": "canonical_evidence_atoms", "page_id": "page:0002"},
        ],
        "evidence_ids": ["p1-start", "p1-end", "p2-start", "p2-end"],
        "field_name": "query_period",
        "derivation": "source_period_envelope",
        "normalized_only": True,
        "source_components": [
            {
                "page_id": "page:0001",
                "raw_name": "起始日期/截止日期",
                "raw_start_name": "起始日期",
                "raw_start": "20230601",
                "raw_end_name": "截止日期",
                "raw_end": "20230630",
                "normalized_start": "2023-06-01",
                "normalized_end": "2023-06-30",
                "evidence_ids": ["p1-start", "p1-end"],
                "source": "canonical_evidence_atoms",
            },
            {
                "page_id": "page:0002",
                "raw_name": "起始日期/截止日期",
                "raw_start_name": "起始日期",
                "raw_start": "20230701",
                "raw_end_name": "截止日期",
                "raw_end": "20230731",
                "normalized_start": "2023-07-01",
                "normalized_end": "2023-07-31",
                "evidence_ids": ["p2-start", "p2-end"],
                "source": "canonical_evidence_atoms",
            },
        ],
    }

    projection, bundle, payload = _project_synthetic_result(monkeypatch, parse_result, result)
    projected_header = projection.datasets["statement_header"][0]
    header = payload["datasets"][0]["rows"][0]

    assert "query_period" not in projected_header["canonical_raw"]
    assert "query_period" not in projected_header["raw"]
    assert header["normalized"]["query_period"] == "2023-06-01 ~ 2023-07-31"
    assert header["normalized"]["period_start"] == "2023-06-01"
    assert header["normalized"]["period_end"] == "2023-07-31"
    internal_header = bundle.semantic_payload()["datasets"][0]["rows"][0]
    assert "raw" not in header and "canonical_raw" not in header
    assert internal_header["canonical_raw"]["period_start"] == "20230601"
    assert internal_header["canonical_raw"]["period_end"] == "20230731"
    assert "query_period" not in internal_header["canonical_raw"]
    assert "query_period" not in internal_header["raw"]
    assert "source period components" not in internal_header["raw"]
    assert json.loads(internal_header["raw"]["起始日期"]) == [
        {"page": 1, "value": "20230601"},
        {"page": 2, "value": "20230701"},
    ]
    assert json.loads(internal_header["raw"]["截止日期"]) == [
        {"page": 1, "value": "20230630"},
        {"page": 2, "value": "20230731"},
    ]
    period_source = internal_header["source"]["field_sources"]["query_period"]
    assert period_source["source"] == "canonical_evidence_atoms"
    assert period_source["derivation"] == "source_period_envelope"
    assert period_source["normalized_only"] is True
    assert period_source["component_count"] == 2
    assert period_source["evidence_ids"] == ["p1-start", "p1-end", "p2-start", "p2-end"]
    assert internal_header["source"]["field_sources"]["period_start"]["evidence_ids"] == ["p1-start", "p1-end"]
    assert internal_header["source"]["field_sources"]["period_end"]["evidence_ids"] == ["p2-start", "p2-end"]
    assert validate_projection_payload("community", payload).valid
    assert bundle.conservation_issues(payload=payload) == []


def test_promoted_signed_transaction_preserves_business_fields_and_source(monkeypatch) -> None:
    parse_result = _parse_result()
    result = _synthetic_extract_result()
    source = {
        "source": "promoted_data_row_table",
        "source_page": 1,
        "page_id": "page:0001",
        "page_range": [1, 1],
        "table_id": "geo_table_0",
        "source_row_index": -1,
        "source_row_role": "promoted_header",
        "reconstructed_row_index": 0,
        "header_source": "data_row",
        "evidence_ids": ["ev:promoted-row"],
        "source_refs": [
            {
                "source": "canonical_page_text",
                "source_page": 1,
                "page_range": [1, 1],
                "bbox": [20.0, 120.0, 560.0, 138.0],
                "evidence_ids": ["ev:promoted-row"],
            }
        ],
    }
    result.records = [
        {
            "record_id": "records:r000001",
            "raw": {
                "日期": "20231007",
                "业务类型": "商户清算",
                "票据号": "20231007C106320166",
                "摘要": "清算入账",
                "借方/贷方金额": "-25.00",
                "余额": "1,547.50",
                "对手户名": "上海测试科技有限公司",
            },
            "canonical_raw": {
                "date": "20231007",
                "transaction_name": "商户清算",
                "voucher_number": "20231007C106320166",
                "summary": "清算入账",
                "amount": "-25.00",
                "balance": "1,547.50",
                "counter_party": "上海测试科技有限公司",
            },
            "normalized": {
                "date": "2023-10-07",
                "transaction_name": "商户清算",
                "voucher_number": "20231007C106320166",
                "summary": "清算入账",
                "amount": 25.0,
                "balance": 1547.5,
                "direction": "expense",
                "counter_party": "上海测试科技有限公司",
            },
            "source": source,
        }
    ]

    projection, bundle, payload = _project_synthetic_result(monkeypatch, parse_result, result)
    projected_transaction = projection.datasets["records"][0]
    transaction = payload["datasets"][1]["rows"][0]
    internal_transaction = bundle.semantic_payload()["datasets"][1]["rows"][0]

    assert projected_transaction["normalized"]["amount"] == 25.0
    assert projected_transaction["normalized"]["direction"] == "expense"
    assert "raw" not in transaction and "canonical_raw" not in transaction
    assert internal_transaction["raw"]["业务类型"] == "商户清算"
    assert internal_transaction["raw"]["票据号"] == "20231007C106320166"
    assert internal_transaction["raw"]["借方/贷方金额"] == "-25.00"
    assert internal_transaction["raw"]["对手户名"] == "上海测试科技有限公司"
    assert internal_transaction["canonical_raw"]["transaction_name"] == "商户清算"
    assert internal_transaction["canonical_raw"]["voucher_number"] == "20231007C106320166"
    assert internal_transaction["canonical_raw"]["amount"] == "-25.00"
    assert internal_transaction["canonical_raw"]["counter_party"] == "上海测试科技有限公司"
    assert transaction["normalized"]["transaction_name"] == "商户清算"
    assert transaction["normalized"]["voucher_number"] == "20231007C106320166"
    assert transaction["normalized"]["amount"] == "25.0"
    assert transaction["normalized"]["direction"] == "expense"
    assert transaction["normalized"]["counter_party"] == "上海测试科技有限公司"
    for key, value in source.items():
        assert projected_transaction["source"][key] == value
        assert internal_transaction["source"][key] == value
    assert validate_projection_payload("community", payload).valid
    assert bundle.conservation_issues(payload=payload) == []


def test_attach_statement_context_fills_empty_values_without_overwriting_source_row() -> None:
    header_records = [
        {
            "record_id": "statement_header:r000001",
            "normalized": {
                "statement_title": "活期账户明细查询",
                "account_holder": "表头户名",
                "account_number": "HEADER-ACCOUNT",
                "currency": "CNY",
                "period_start": "2023-02-23",
                "period_end": "2023-05-22",
                "print_date": "2023-08-24",
            },
            "canonical_raw": {},
            "raw": {},
            "source": {"page_range": [1, 5], "field_sources": {}},
        }
    ]
    source_record = _transaction_record(own_account="ROW-ACCOUNT", currency="USD")
    source_record["normalized"]["account_holder"] = "逐行户名"

    attached = attach_statement_context([source_record], header_records)
    normalized = attached[0]["normalized"]

    assert normalized["statement_header_id"] == "statement_header:r000001"
    assert normalized["own_account"] == "ROW-ACCOUNT"
    assert normalized["currency"] == "USD"
    assert normalized["account_holder"] == "逐行户名"
    assert normalized["statement_title"] == "活期账户明细查询"
    assert normalized["period_start"] == "2023-02-23"
    assert normalized["period_end"] == "2023-05-22"
    assert normalized["print_date"] == "2023-08-24"
    assert source_record["normalized"].get("statement_header_id") is None


def test_filename_only_institution_is_not_published_as_business_issuer(monkeypatch) -> None:
    parse_result = _parse_result()
    result = _synthetic_extract_result()
    result.style_meta.institution_hint = "中国建设银行"
    result.style_meta.institution_authority = "filename.token"
    monkeypatch.setattr(community_module, "run_bank_statement_extract", lambda *_args: result)
    monkeypatch.setattr(
        community_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence.empty(),
    )

    projection = BankStatementCommunityPlugin().derive(parse_result, "")

    assert "institution_hint" not in projection.domain_facts
    assert "institution_authority" not in projection.domain_facts
    assert "organization" not in projection.entity_fields
    assert "bank_name" not in projection.domain_facts["field_details"]
    assert all("bank_name" not in row.get("normalized", {}) for row in projection.datasets["statement_header"])
    assert "中国建设银行" not in projection.content_markdown_override


def test_explicit_source_issuer_overrides_internal_routing_hint(monkeypatch) -> None:
    parse_result = _parse_result()
    result = _synthetic_extract_result()
    result.identity_fields["bank_name"] = _detail("银行名称", "测试银行")
    result.style_meta.institution_hint = "错误路由银行"
    result.style_meta.institution_authority = "filename.token"
    monkeypatch.setattr(community_module, "run_bank_statement_extract", lambda *_args: result)
    monkeypatch.setattr(
        community_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence.empty(),
    )

    projection = BankStatementCommunityPlugin().derive(parse_result, "")

    assert projection.domain_facts["institution_hint"] == "测试银行"
    assert projection.domain_facts["institution_authority"] == "identity.bank_name"
    assert projection.entity_fields["organization"] == "测试银行"
    assert projection.datasets["statement_header"][0]["normalized"]["bank_name"] == "测试银行"
    assert "错误路由银行" not in projection.content_markdown_override


def test_digital_bank_enables_compact_export_without_pruning_extracted_records(monkeypatch) -> None:
    projection, bundle, payload = _project_synthetic_result(
        monkeypatch, _parse_result(), _synthetic_extract_result()
    )
    assert bundle.compact_output["omit_absent_fields"] is True
    assert bundle.compact_output["minify_json"] is True
    assert bundle.compact_output["normalized_only"] is True
    assert bundle.compact_output["business_view"] is True
    assert payload["schema"]["version"] == "5.0.0"
    header = payload["datasets"][0]
    assert "omitted_normalized_fields" not in header
    assert "wechat_id" not in header["rows"][0]["normalized"]
    assert "wechat_id" not in {column["key"] for column in header["columns"]}
    assert "wechat_id" in {column["key"] for column in bundle.datasets[0].public["columns"]}
    assert bundle.datasets[0].rows == projection.datasets["statement_header"]
    assert "account_number" in header["rows"][0]["normalized"]


def test_scanned_bank_keeps_existing_dense_export_default(monkeypatch) -> None:
    result = _synthetic_extract_result()
    result.extraction_route = BankExtractionRoute.SCANNED
    _projection, bundle, payload = _project_synthetic_result(monkeypatch, _parse_result(), result)
    assert bundle.compact_output == {}
    assert "omitted_normalized_fields" not in payload["datasets"][0]
    assert payload["datasets"][0]["rows"][0]["normalized"]["wechat_id"] is None
