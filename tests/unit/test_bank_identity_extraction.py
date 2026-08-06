# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank statement identity field extraction tests."""

from __future__ import annotations

from types import SimpleNamespace

from docmirror.models.entities.parse_result import DocumentEntities, KeyValuePair, PageContent, ParseResult, TextBlock
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.extract_pipeline import enrich_identity_fields


def test_extract_identity_matches_账户号_kv():
    plugin = BankStatementCommunityPlugin()
    pr = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                key_values=[KeyValuePair(key="账户号", value="03-869900040010370")],
            )
        ]
    )
    fields = plugin._extract_identity(pr)
    assert fields["account_number"]["normalized_value"] == "03-869900040010370"


def test_enrich_identity_maps_subject_id_to_account_number():
    pr = ParseResult(
        entities=DocumentEntities(
            organization="中国农业银行",
            subject_id="03-869900040010370",
        )
    )
    fields = enrich_identity_fields({}, "", pr, institution="中国农业银行")
    assert fields["bank_name"]["normalized_value"] == "中国农业银行"
    assert fields["account_number"]["normalized_value"] == "03-869900040010370"


def test_extract_identity_rejects_transaction_summary_account_kv() -> None:
    plugin = BankStatementCommunityPlugin()
    pr = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                key_values=[KeyValuePair(key="收息，结息账号", value="999019305110001")],
            )
        ]
    )

    assert "account_number" not in plugin._extract_identity(pr)


def test_enrich_identity_rejects_body_subject_id_without_header_account() -> None:
    text = (
        "交易日期 借方(出账) 贷方(入账) 余额 摘要 收(付)方名称 收(付)方账号 交易类型\n"
        "2025-03-21 31.55 27,486.46 收息，结息账号:999019305110001\n"
    )
    pr = ParseResult(entities=DocumentEntities(subject_id="99901930511000"))

    fields = enrich_identity_fields({}, text, pr)

    assert "account_number" not in fields


def test_header_identity_normalizes_cjk_compatibility_glyphs_before_matching():
    fields = enrich_identity_fields(
        {},
        (
            "中国农业银⾏账⼾活期交易明细清单\n"
            "⼾名：测试用户  账⼾：6230****6516  币种：⼈⺠币\n"
            "起⽌⽇期：20220808 至 20220908\n"
            "交易⽇期 交易⾦额 账户余额"
        ),
    )

    assert fields["account_holder"]["normalized_value"] == "测试用户"
    assert fields["account_number"]["normalized_value"] == "6230****6516"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["query_period"]["normalized_value"] == "2022-08-08 ~ 2022-09-08"


def test_evidence_identity_aggregates_explicit_periods_across_pages() -> None:
    atoms: list[dict] = []
    for page_number, start, end in (
        (1, "20230601", "20230630"),
        (2, "20230701", "20230731"),
        (3, "20230801", "20230831"),
    ):
        page_id = f"page:{page_number:04d}"
        atoms.extend(
            [
                {
                    "id": f"start-{page_number}",
                    "page_id": page_id,
                    "text": f"起始日期:{start}",
                    "bbox": [20.0, 35.0, 130.0, 43.0],
                },
                {
                    "id": f"end-{page_number}",
                    "page_id": page_id,
                    "text": f"截止日期:{end}",
                    "bbox": [150.0, 35.0, 260.0, 43.0],
                },
                {
                    "id": f"date-{page_number}",
                    "page_id": page_id,
                    "text": "记账日期",
                    "bbox": [50.0, 80.0, 95.0, 88.0],
                },
                {
                    "id": f"amount-{page_number}",
                    "page_id": page_id,
                    "text": "交易金额",
                    "bbox": [150.0, 80.0, 195.0, 88.0],
                },
                {
                    "id": f"balance-{page_number}",
                    "page_id": page_id,
                    "text": "账户余额",
                    "bbox": [230.0, 80.0, 275.0, 88.0],
                },
            ]
        )
    parse_result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={"text_atoms": atoms}, pages=[]),
        parser_info=SimpleNamespace(options={}),
    )

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(parse_result)

    assert fields["query_period"]["normalized_value"] == "2023-06-01 至 2023-08-31"
    assert [ref["page_id"] for ref in fields["query_period"]["source_refs"]] == [
        "page:0001",
        "page:0002",
        "page:0003",
    ]


def test_identity_enrichment_aggregates_explicit_page_header_periods() -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="起始日期:20230601\n截止日期:20230630\n序号 记账日期 交易金额 账户余额")],
            ),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="起始日期:20230701\n截止日期:20230731\n序号 记账日期 交易金额 账户余额")],
            ),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="起始日期:20230801\n截止日期:20230831\n序号 记账日期 交易金额 账户余额")],
            ),
        ]
    )

    fields = enrich_identity_fields({}, "起始日期:20230601\n截止日期:20230630", parse_result)

    assert fields["query_period"]["normalized_value"] == "2023-06-01 ~ 2023-08-31"
    assert [ref["source_page"] for ref in fields["query_period"]["source_refs"]] == [1, 2, 3]
