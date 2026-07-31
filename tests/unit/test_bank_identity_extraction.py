# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank statement identity field extraction tests."""

from __future__ import annotations

from docmirror.models.entities.parse_result import DocumentEntities, KeyValuePair, PageContent, ParseResult
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
