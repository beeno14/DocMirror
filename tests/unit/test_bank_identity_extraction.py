# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank statement identity field extraction tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, KeyValuePair, PageContent, ParseResult, TextBlock
from docmirror.plugins.bank_statement import community_plugin as community_module
from docmirror.plugins.bank_statement import statement_context
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.extract_pipeline import enrich_identity_fields
from docmirror.plugins.bank_statement.institution_authority import extract_identity_from_header


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


def test_header_identity_does_not_treat_counterparty_column_as_account_holder() -> None:
    text = """
    账务明细清单
    账号: 591907551610902
    账户名称: 福州北辰星汽车服务有限公司
    上页余额: 1,572.50
    日期
    业务类型
    票据号
    摘要
    借方/贷方金额
    余额
    对手户名
    20231017
    支付平台退回：账号、户名不符
    """

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "福州北辰星汽车服务有限公司"


def test_enrich_identity_keeps_routing_institution_out_of_bank_name():
    pr = ParseResult(
        entities=DocumentEntities(
            organization="中国农业银行",
            subject_id="03-869900040010370",
        )
    )
    fields = enrich_identity_fields({}, "", pr, institution="中国农业银行")
    assert "bank_name" not in fields
    assert fields["account_number"]["normalized_value"] == "03-869900040010370"


def test_enrich_identity_rejects_filename_only_issuer() -> None:
    parse_result = SimpleNamespace(
        pages=[],
        entities=DocumentEntities(),
        file_path="/tmp/银行流水_中国建设银行_20231228.pdf",
    )

    fields = enrich_identity_fields({}, "网上银行 网银结算", parse_result)

    assert "bank_name" not in fields


def test_enrich_identity_separates_explicit_issuer_from_opening_branch() -> None:
    parse_result = SimpleNamespace(pages=[], entities=DocumentEntities(), file_path="/tmp/statement.pdf")
    fields = enrich_identity_fields(
        {},
        "银行名称：测试银行\n开户行：测试银行科技支行\n交易日期 交易金额 余额",
        parse_result,
    )

    assert fields["bank_name"]["normalized_value"] == "测试银行"
    assert fields["bank_name"]["raw_name"] == "银行名称"
    assert fields["bank_name"]["source"] == "header.kv"
    assert fields["branch_name"]["normalized_value"] == "测试银行科技支行"
    assert fields["branch_name"]["raw_name"] == "开户行"


def test_opening_branch_detail_cannot_be_relabelled_as_bank_name() -> None:
    fields = enrich_identity_fields(
        {
            "bank_name": {
                "raw_name": "开户行",
                "raw_value": "测试银行科技支行",
                "normalized_value": "测试银行科技支行",
                "source_refs": [{"source": "canonical_evidence_atoms", "page": 1}],
            }
        },
        "",
        SimpleNamespace(pages=[], entities=DocumentEntities()),
    )

    assert "bank_name" not in fields


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
                    "id": f"start-label-{page_number}",
                    "page_id": page_id,
                    "text": "起始日期:",
                    "bbox": [20.0, 35.0, 65.0, 43.0],
                },
                {
                    "id": f"start-value-{page_number}",
                    "page_id": page_id,
                    "text": start,
                    "bbox": [70.0, 35.0, 130.0, 43.0],
                },
                {
                    "id": f"end-label-{page_number}",
                    "page_id": page_id,
                    "text": "截止日期:",
                    "bbox": [150.0, 35.0, 195.0, 43.0],
                },
                {
                    "id": f"end-value-{page_number}",
                    "page_id": page_id,
                    "text": end,
                    "bbox": [200.0, 35.0, 260.0, 43.0],
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

    period = fields["query_period"]
    assert period["normalized_value"] == "2023-06-01 至 2023-08-31"
    assert "raw_value" not in period
    assert period["derivation"] == "source_period_envelope"
    assert period["normalized_only"] is True
    assert [ref["page_id"] for ref in fields["query_period"]["source_refs"]] == [
        "page:0001",
        "page:0002",
        "page:0003",
    ]
    assert [component["raw_start"] for component in period["source_components"]] == [
        "20230601",
        "20230701",
        "20230801",
    ]
    assert [component["raw_end"] for component in period["source_components"]] == [
        "20230630",
        "20230731",
        "20230831",
    ]
    assert period["source_components"][0]["evidence_ids"] == [
        "start-label-1",
        "start-value-1",
        "end-label-1",
        "end-value-1",
    ]

    facts = statement_context._identity_facts({"query_period": period})
    record = statement_context._record_from_facts(facts, [1, 2, 3], 1)

    assert record["normalized"]["query_period"] == "2023-06-01 ~ 2023-08-31"
    assert "query_period" not in record["canonical_raw"]
    assert record["canonical_raw"]["period_start"] == "20230601"
    assert record["canonical_raw"]["period_end"] == "20230831"
    assert record["raw"]["起始日期"] == [
        {"page": 1, "value": "20230601"},
        {"page": 2, "value": "20230701"},
        {"page": 3, "value": "20230801"},
    ]
    period_source = record["source"]["field_sources"]["query_period"]
    assert period_source["derivation"] == "source_period_envelope"
    assert period_source["component_count"] == 3
    assert period_source["evidence_ids"] == [
        "start-label-1",
        "start-value-1",
        "end-label-1",
        "end-value-1",
        "start-label-2",
        "start-value-2",
        "end-label-2",
        "end-value-2",
        "start-label-3",
        "start-value-3",
        "end-label-3",
        "end-value-3",
    ]
    assert record["source"]["field_sources"]["period_start"]["evidence_ids"] == [
        "start-label-1",
        "start-value-1",
        "end-label-1",
        "end-value-1",
    ]
    assert record["source"]["field_sources"]["period_end"]["evidence_ids"] == [
        "start-label-3",
        "start-value-3",
        "end-label-3",
        "end-value-3",
    ]


@pytest.mark.parametrize(
    ("label", "raw_start", "separator", "raw_end"),
    [
        ("账单所属期间", "20240201", "至", "20240229"),
        ("Statement Covered Period", "2024/02/01", "~", "2024/02/29"),
    ],
    ids=["chinese", "english"],
)
def test_evidence_labelled_single_period_alias_projects_exact_source_pair(
    label: str,
    raw_start: str,
    separator: str,
    raw_end: str,
) -> None:
    atoms = [
        {"id": "period-label", "text": f"{label}:", "bbox": [10.0, 20.0, 140.0, 30.0]},
        {"id": "period-start", "text": raw_start, "bbox": [145.0, 20.0, 215.0, 30.0]},
        {"id": "period-separator", "text": separator, "bbox": [220.0, 20.0, 230.0, 30.0]},
        {"id": "period-end", "text": raw_end, "bbox": [235.0, 20.0, 305.0, 30.0]},
    ]

    period = community_module._evidence_document_query_period({"page:0001": atoms})

    assert period is not None
    assert period["raw_name"] == label
    assert period["normalized_value"] == "2024-02-01 至 2024-02-29"
    assert period["derivation"] == "source_period_envelope"
    assert period["normalized_only"] is True
    assert period["source_components"] == [
        {
            "page_id": "page:0001",
            "raw_name": label,
            "raw_start_name": label,
            "raw_start": raw_start,
            "raw_end_name": label,
            "raw_end": raw_end,
            "normalized_start": "2024-02-01",
            "normalized_end": "2024-02-29",
            "evidence_ids": ["period-label", "period-start", "period-separator", "period-end"],
            "source": "canonical_evidence_atoms",
        }
    ]

    facts = statement_context._identity_facts({"query_period": period})
    record = statement_context._record_from_facts(facts, [1], 1)

    assert record["normalized"]["query_period"] == "2024-02-01 ~ 2024-02-29"
    assert record["canonical_raw"]["period_start"] == raw_start
    assert record["canonical_raw"]["period_end"] == raw_end
    assert "query_period" not in record["canonical_raw"]
    assert record["raw"][label] == [
        {"page": 1, "value": raw_start},
        {"page": 1, "value": raw_end},
    ]


def test_evidence_year_month_period_projects_exact_leap_year_components() -> None:
    atoms = [
        {"id": "year-label", "text": "年份:", "bbox": [10.0, 20.0, 50.0, 30.0]},
        {"id": "year-value", "text": "2024", "bbox": [55.0, 20.0, 85.0, 30.0]},
        {"id": "month-label", "text": "月份:", "bbox": [95.0, 20.0, 135.0, 30.0]},
        {"id": "month-value", "text": "2", "bbox": [140.0, 20.0, 150.0, 30.0]},
    ]

    period = community_module._evidence_year_month_query_period(atoms, "page:0001")

    assert period is not None
    assert period["normalized_value"] == "2024-02-01 至 2024-02-29"
    assert period["derivation"] == "source_year_month_period"
    assert period["normalized_only"] is True
    assert period["source_components"] == [
        {
            "page_id": "page:0001",
            "raw_year_name": "年份",
            "raw_year": "2024",
            "raw_month_name": "月份",
            "raw_month": "2",
            "evidence_ids": ["year-label", "year-value", "month-label", "month-value"],
            "source": "canonical_evidence_atoms",
        }
    ]

    facts = statement_context._identity_facts({"query_period": period})
    record = statement_context._record_from_facts(facts, [1], 1)

    assert record["normalized"]["statement_year"] == 2024
    assert record["normalized"]["statement_month_number"] == 2
    assert record["normalized"]["statement_month"] == "2024-02"
    assert record["normalized"]["query_period"] == "2024-02-01 ~ 2024-02-29"
    assert record["canonical_raw"]["statement_year"] == "2024"
    assert record["canonical_raw"]["statement_month_number"] == "2"
    assert "query_period" not in record["canonical_raw"]
    assert record["source"]["field_sources"]["statement_year"]["evidence_ids"] == [
        "year-label",
        "year-value",
        "month-label",
        "month-value",
    ]


def _source_period_detail(components: list[tuple[int, str, str]]) -> dict:
    starts = [start for _page, start, _end in components]
    ends = [end for _page, _start, end in components]
    return {
        "normalized_value": f"{min(starts)} 至 {max(ends)}",
        "source": "canonical_evidence_atoms",
        "derivation": "source_period_envelope",
        "normalized_only": True,
        "source_components": [
            {
                "page_id": f"page:{page:04d}",
                "raw_name": "起始日期/截止日期",
                "raw_start_name": "起始日期",
                "raw_start": start.replace("-", ""),
                "raw_end_name": "截止日期",
                "raw_end": end.replace("-", ""),
                "evidence_ids": [f"period:{page}:{index}:start", f"period:{page}:{index}:end"],
                "source": "canonical_evidence_atoms",
            }
            for index, (page, start, end) in enumerate(components)
        ],
    }


@pytest.mark.parametrize(
    "components",
    [
        [(1, "2024-01-01", "2024-01-31"), (2, "2024-01-31", "2024-02-29")],
        [(1, "2024-01-01", "2024-01-31"), (2, "2024-02-02", "2024-02-29")],
        [(1, "2024-02-01", "2024-02-29"), (2, "2024-01-01", "2024-01-31")],
        [(1, "2024-01-01", "2024-01-31"), (1, "2024-02-01", "2024-02-29")],
    ],
    ids=["overlap", "gap", "reordered", "duplicate-page"],
)
def test_source_period_envelope_rejects_incoherent_single_scope_components(
    components: list[tuple[int, str, str]],
) -> None:
    period = _source_period_detail(components)

    assert statement_context._identity_facts({"query_period": period}) == []


def test_source_period_envelope_is_recomputed_inside_each_statement_scope(monkeypatch) -> None:
    period = {
        "normalized_value": "2023-06-01 至 2023-07-31",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {"source": "canonical_evidence_atoms", "page_id": "page:0001"},
            {"source": "canonical_evidence_atoms", "page_id": "page:0002"},
        ],
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
                "evidence_ids": ["p2-start", "p2-end"],
                "source": "canonical_evidence_atoms",
            },
        ],
    }
    page_facts = {
        page: [
            statement_context._HeaderFact(
                "account_number",
                "账号",
                str(page),
                str(page),
                page,
                f"page:{page:04d}",
                None,
                (f"account-{page}",),
            )
        ]
        for page in (1, 2)
    }
    monkeypatch.setattr(statement_context, "_page_header_facts", lambda _result: (page_facts, {}))
    monkeypatch.setattr(statement_context, "_context_page_groups", lambda _result, _facts: [[1], [2]])

    records = statement_context.build_statement_header_records(
        SimpleNamespace(pages=[SimpleNamespace(page_number=1), SimpleNamespace(page_number=2)]),
        {"query_period": period},
    )

    assert [record["normalized"]["query_period"] for record in records] == [
        "2023-06-01 ~ 2023-06-30",
        "2023-07-01 ~ 2023-07-31",
    ]
    assert [record["canonical_raw"]["period_start"] for record in records] == ["20230601", "20230701"]
    assert [record["canonical_raw"]["period_end"] for record in records] == ["20230630", "20230731"]
    assert [record["source"]["field_sources"]["query_period"]["component_count"] for record in records] == [1, 1]


def test_source_period_component_outside_resolved_statement_scopes_cannot_leak(monkeypatch) -> None:
    period = _source_period_detail(
        [
            (1, "2024-01-01", "2024-01-31"),
            (3, "2024-03-01", "2024-03-31"),
        ]
    )
    page_facts = {
        page: [
            statement_context._HeaderFact(
                "account_number",
                "账号",
                f"622200000000000{page}",
                f"622200000000000{page}",
                page,
                f"page:{page:04d}",
                None,
                (f"account-{page}",),
            )
        ]
        for page in (1, 2)
    }
    monkeypatch.setattr(statement_context, "_page_header_facts", lambda _result: (page_facts, {}))
    monkeypatch.setattr(statement_context, "_context_page_groups", lambda _result, _facts: [[1], [2]])

    records = statement_context.build_statement_header_records(
        SimpleNamespace(pages=[SimpleNamespace(page_number=1), SimpleNamespace(page_number=2)]),
        {"query_period": period},
    )

    assert records[0]["normalized"]["query_period"] == "2024-01-01 ~ 2024-01-31"
    assert "query_period" not in records[1]["normalized"]
    assert "period_start" not in records[1]["normalized"]
    assert "period_end" not in records[1]["normalized"]
    assert all("202403" not in str(record["raw"]) for record in records)


def test_evidence_period_rejects_geometrically_distant_label_value_atoms() -> None:
    atoms = [
        {"id": "start-label", "text": "起始日期:", "bbox": [10.0, 10.0, 60.0, 20.0]},
        {"id": "start-value", "text": "20230101", "bbox": [70.0, 300.0, 130.0, 310.0]},
        {"id": "end-label", "text": "截止日期:", "bbox": [10.0, 500.0, 60.0, 510.0]},
        {"id": "end-value", "text": "20230131", "bbox": [70.0, 700.0, 130.0, 710.0]},
    ]

    assert community_module._evidence_document_query_period({"page:0001": atoms}) is None


def test_evidence_period_rejects_components_without_atom_ids() -> None:
    atoms = [
        {"text": "起始日期:", "bbox": [10.0, 10.0, 60.0, 20.0]},
        {"text": "20230101", "bbox": [70.0, 10.0, 130.0, 20.0]},
        {"text": "截止日期:", "bbox": [150.0, 10.0, 200.0, 20.0]},
        {"text": "20230131", "bbox": [210.0, 10.0, 270.0, 20.0]},
    ]

    assert community_module._evidence_document_query_period({"page:0001": atoms}) is None


def test_evidence_year_month_rejects_distant_or_unidentified_atoms() -> None:
    distant = [
        {"id": "year-label", "text": "年份:", "bbox": [10.0, 10.0, 50.0, 20.0]},
        {"id": "year-value", "text": "2025", "bbox": [60.0, 300.0, 90.0, 310.0]},
        {"id": "month-label", "text": "月份:", "bbox": [100.0, 500.0, 140.0, 510.0]},
        {"id": "month-value", "text": "07", "bbox": [150.0, 700.0, 170.0, 710.0]},
    ]
    unidentified = [
        {"text": "年份:", "bbox": [10.0, 10.0, 50.0, 20.0]},
        {"text": "2025", "bbox": [60.0, 10.0, 90.0, 20.0]},
        {"text": "月份:", "bbox": [100.0, 10.0, 140.0, 20.0]},
        {"text": "07", "bbox": [150.0, 10.0, 170.0, 20.0]},
    ]

    assert community_module._evidence_year_month_query_period(distant, "page:0001") is None
    assert community_module._evidence_year_month_query_period(unidentified, "page:0001") is None


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
    assert "raw_value" not in fields["query_period"]
    assert fields["query_period"]["derivation"] == "source_period_envelope"
    assert [component["raw_start"] for component in fields["query_period"]["source_components"]] == [
        "20230601",
        "20230701",
        "20230801",
    ]
    assert [ref["source_page"] for ref in fields["query_period"]["source_refs"]] == [1, 2, 3]
