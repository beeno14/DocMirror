# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for bank statement header/context Community projection."""

from __future__ import annotations

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
    assert header["raw"]["币种"] == "人民币"
    assert header["canonical_raw"]["currency"] == "人民币"
    assert header["source"]["field_sources"]["account_number"]["evidence_ids"] == [
        "header:p1:账号"
    ]
    assert "style_id" not in header["normalized"]
    assert "extraction_route" not in header["normalized"]

    transaction = payload["datasets"][1]["rows"][0]["normalized"]
    assert payload["datasets"][1]["foreign_keys"] == [
        {
            "columns": ["statement_header_id"],
            "reference_dataset": "statement_header",
            "reference_columns": ["record_id"],
        }
    ]
    assert transaction["statement_header_id"] == header["record_id"]
    assert transaction["own_account"] == "3211020801201000170968"
    assert transaction["account_holder"] == "测试工具有限公司"
    assert transaction["currency"] == "CNY"
    assert transaction["statement_title"] == "活期账户明细查询"
    assert transaction["period_start"] == "2023-02-23"
    assert transaction["period_end"] == "2023-05-22"
    assert transaction["print_date"] == "2023-08-24"
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
