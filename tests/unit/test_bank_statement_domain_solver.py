# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from calendar import monthrange
from types import SimpleNamespace

import pytest

from docmirror.plugins.bank_statement.canonical_quality import audit_cqf
from docmirror.plugins.bank_statement.extract_pipeline import enrich_identity_fields
from docmirror.plugins.bank_statement.institution import detect_registered_institution
from docmirror.plugins.bank_statement.institution_authority import (
    extract_identity_from_header,
    resolve_institution_from_context,
)
from docmirror.plugins.bank_statement.semantic_solver import (
    BankStatementSemanticSolver,
    _extract_counterparty,
)
from docmirror.plugins.bank_statement.semantic_solver import extract_identity as extract_solver_identity
from docmirror.plugins.bank_statement.wide_table_recovery import (
    RowCountEvidence,
    audit_bank_statement_invariants,
    count_expected_rows_from_bank_footer,
    page_texts_from_parse_result,
    resolve_row_count_evidence,
)

VERTICAL_LEDGER_TEXT = """
江苏银行对公账户对账单
账户名称：测试有限公司
账号：70650188000156836
起始日期：2022-01-01 终止日期：2022-01-31
借方笔数：2
借方发生总额：12.00
贷方笔数：1
贷方发生总额：5.00
合计笔数：3
1
2022-01-01
09:00:00
往来款
10.00
90.00
1104010309000388824
张三有限公司
2
2022-01-02
10:00:00
货款
5.00
95.00
2204010309000388825
李四有限公司
3
2022-01-03
11:00:00
收费
2.00
93.00
70650107360000033
企业电子渠道跨行转账手续费收入
""".strip()


def test_bank_statement_solver_reconciles_vertical_debit_credit_ledger() -> None:
    solution = BankStatementSemanticSolver().solve(full_text=VERTICAL_LEDGER_TEXT)

    assert solution.success
    assert {item["id"]: item["status"] for item in solution.invariant_results} == {
        "bank.row_count_reconciliation": "pass",
        "bank.debit_credit_count_reconciliation": "pass",
        "bank.debit_credit_total_reconciliation": "pass",
        "bank.balance_chain_consistency": "pass",
    }

    model = solution.canonical_model
    records = model["records"]
    assert [record["direction"] for record in records] == ["expense", "income", "expense"]
    assert [record["amount"] for record in records] == [-10.0, 5.0, -2.0]
    assert records[0]["timestamp"] == "2022-01-01T09:00:00"
    assert records[0]["counter_account"] == "1104010309000388824"
    assert records[0]["counter_party"] == "张三有限公司"
    assert model["identity"]["account_holder"] == "测试有限公司"
    assert model["identity"]["query_period"] == "2022-01-01 ~ 2022-01-31"
    assert model["identity"]["bank_name"] == "江苏银行"

    split_table = model["split_table"]
    assert split_table[0] == [
        "序号",
        "交易日期",
        "交易时间",
        "摘要",
        "借方发生额",
        "贷方发生额",
        "余额",
        "对方账户",
        "对方户名",
    ]
    assert split_table[1][4] == "10.00"
    assert split_table[2][5] == "5.00"
    assert split_table[3][4] == "2.00"


def test_bank_statement_solver_does_not_take_over_without_debit_credit_header() -> None:
    solution = BankStatementSemanticSolver().solve(
        full_text="""
        交易明细
        2024-01-01 支付宝 10.00 支出 90.00
        2024-01-02 工资 20.00 收入 110.00
        """,
    )

    assert not solution.success
    assert solution.status == "failed"
    assert solution.diagnostics == ({"reason": "missing_debit_credit_header_totals"},)


def test_semantic_identity_keeps_global_bank_mentions_out_of_issuer() -> None:
    identity = extract_solver_identity(
        "账户名称：测试企业\n开户机构：测试银行科技支行\n对方户名：中国建设银行股份有限公司"
    )

    assert "bank_name" not in identity
    assert identity["branch_name"] == "测试银行科技支行"


def test_vertical_header_identity_overrides_weak_transaction_summary_identity() -> None:
    text = """
    江苏银行对公账户对账单（本对账单仅供参考）
    起始日期：2022-06-01
    2022-08-31
    终止日期：
    镇江一生一世好游戏有限公司
    账户名称：
    70650188000156836
    账号：
    借方笔数：61
    贷方笔数：30
    序号
    摘要
    往来款
    """.strip()

    assert extract_identity_from_header(text) == {
        "account_holder": "镇江一生一世好游戏有限公司",
        "account_number": "70650188000156836",
        "query_period": "2022-06-01 ~ 2022-08-31",
    }

    fields = enrich_identity_fields(
        {
            "account_holder": {
                "raw_name": "account_holder",
                "raw_value": "往来款",
                "normalized_value": "往来款",
                "data_type": "string",
            },
        },
        text,
    )

    assert fields["account_holder"]["normalized_value"] == "镇江一生一世好游戏有限公司"
    assert fields["account_number"]["normalized_value"] == "70650188000156836"
    assert "currency" not in fields


def test_currency_is_recovered_from_an_explicit_source_table_column() -> None:
    currency_cell = SimpleNamespace(
        text="CNY",
        source_cell_refs=[{"page": 1, "table_id": "pt_1_0", "row": 0, "col": 1}],
        evidence_ids=["ev:currency"],
    )
    parse_result = SimpleNamespace(
        entities=None,
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                tables=[
                    SimpleNamespace(
                        table_id="pt_1_0",
                        headers=["交易日期", "币种"],
                        rows=[
                            SimpleNamespace(
                                cells=[SimpleNamespace(text="2024-01-01"), currency_cell],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    fields = enrich_identity_fields(
        {
            "bank_name": {
                "raw_name": "bank_name",
                "raw_value": "测试银行",
                "normalized_value": "测试银行",
                "data_type": "string",
            }
        },
        "",
        parse_result,
    )

    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["currency"]["source"] == "source_table.currency"
    assert fields["currency"]["evidence_ids"] == ["ev:currency"]


def test_horizontal_header_identity_for_wide_bank_statement() -> None:
    text = """
    交通银行
    交通银行宁夏回族自治区分行明细对账单
    开户机构：交通银行银川开发区支行
    币种：人民币
    年份：2025
    月份：11
    账号： 641301106013000859983 户名： 重庆正大华日软件有限公司银川分公司
    序号 会计日期 交易日期 交易名称 借方发生额 贷方发生额 余额
    """.strip()

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "重庆正大华日软件有限公司银川分公司"
    assert identity["account_number"] == "641301106013000859983"
    assert identity["branch_name"] == "交通银行银川开发区支行"
    assert "bank_name" not in identity
    assert identity["currency"] == "CNY"
    assert identity["query_period"] == "2025-11-01 ~ 2025-11-30"


def test_label_block_header_identity_for_wide_bank_statement() -> None:
    text = """
    交通银行宁夏回族自治区分行明细对账单
    户名：
    页码：
    年份：
    币种：
    账号：
    开户机构：交通银行银川开发区支行
    641301106013000859983
    重庆正大华日软件有限公司银川分公司
    本月第1份-第1页
    2025
    人民币
    88B8563F
    交通银行
    11
    月份：
    借方发生额
    """.strip()

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "重庆正大华日软件有限公司银川分公司"
    assert identity["account_number"] == "641301106013000859983"
    assert identity["branch_name"] == "交通银行银川开发区支行"
    assert "bank_name" not in identity
    assert identity["currency"] == "CNY"
    assert identity["query_period"] == "2025-11-01 ~ 2025-11-30"


def test_ccb_header_totals_and_query_period_are_supported() -> None:
    text = """
    账号： 13355000000062937 账户名称： 镇江海翔机械制造有限公司
    查询日期： 2023-10-01至2023-12-31
    交易日期 交易时间 支出金额 收入金额 余额 对方账号 对方户名 摘要
    收入总金额： 308361.39 收入总笔数： 21
    支出总金额： 310212.14 支出总笔数： 42
    """.strip()

    assert count_expected_rows_from_bank_footer(text) == 63
    identity = extract_identity_from_header(text)
    assert identity["account_holder"] == "镇江海翔机械制造有限公司"
    assert identity["account_number"] == "13355000000062937"
    assert identity["query_period"] == "2023-10-01 ~ 2023-12-31"


def test_split_footer_accepts_total_expense_count_label_with_extra_character() -> None:
    text = """
    账户明细
    账号: 1234567890123456 户名: 测试企业 起止日期: 2025-01-01 - 2025-12-31
    交易日期 交易金额 账户余额 对方账号 摘要
    总收入笔数 222 总收入金额 1770369.54 总支出入笔数 275 总支出金额 1768504.94
    """

    evidence = resolve_row_count_evidence("", page_texts=[(26, text)])

    assert evidence.count == 497
    assert evidence.source == "split_footer"
    assert evidence.confidence == 0.98
    assert evidence.page == 26


def test_bank_header_total_record_count_is_an_independent_expected_count() -> None:
    text = """
    账户交易明细表
    打印日期：2026-02-24
    总条数：38
    序号 交易日期 支出（元） 收入（元） 账户余额（元） 对方账号 对方户名 摘要
    交易时段：2025-01-01 至 2025-12-31
    户名：测试企业
    账号：1234567890123456
    """

    assert count_expected_rows_from_bank_footer(text) == 38


def test_flattened_text_does_not_treat_page_numbers_as_total_rows() -> None:
    text = "\u603b\u7b14\u6570:\n3\n\u603b\u7b14\u6570\n:\n2"

    evidence = resolve_row_count_evidence(text)

    assert evidence.count == 0
    assert evidence.source == "none"


def test_page_local_count_evidence_accepts_a_bounded_total() -> None:
    page = """
    账户名称：测试企业 账号：1234567890123456 起止日期：2025-01-01 - 2025-12-31
    交易日期 交易金额 余额 对方账号 摘要
    交易总金额：900.00 借方累计金额：400.00 贷方累计金额：500.00 总笔数: 90
    """
    evidence = resolve_row_count_evidence("", page_texts=[(1, page)])

    assert evidence.count == 90
    assert evidence.source == "header_total"
    assert evidence.page == 1
    assert evidence.confidence >= 0.9


def test_page_local_count_evidence_accepts_label_value_line_break() -> None:
    page = """
    账户名称：测试企业 账号：1234567890123456 起止日期：2025-01-01 - 2025-12-31
    交易日期 交易金额 余额 对方账号 摘要
    交易总金额：900.00 借方累计金额：400.00 贷方累计金额：500.00
    总笔数:
    90
    """
    evidence = resolve_row_count_evidence("", page_texts=[(1, page)])

    assert evidence.count == 90
    assert evidence.source == "header_total"
    assert evidence.page == 1


def test_terminal_header_total_requires_complete_ordered_page_scopes() -> None:
    contract = """
    账户明细
    账户名称：测试企业
    账号：1234567890123456
    起止日期：2025-01-01 - 2025-12-31
    交易日期 交易金额 余额 对方账号 摘要
    """
    terminal_summary = """
    支出总金额：10.00
    收入总金额：20.00
    总笔数: 7
    """

    accepted = resolve_row_count_evidence("", page_texts=[(1, contract), (2, terminal_summary)])
    malformed_scopes = (
        [(1, contract), (3, terminal_summary)],
        [(1, contract), (1, terminal_summary)],
        [(2, contract), (1, terminal_summary)],
    )

    assert accepted == RowCountEvidence(7, "header_total", 0.94, 2)
    for scopes in malformed_scopes:
        assert resolve_row_count_evidence("", page_texts=scopes).source != "header_total"


def _repeated_current_account_query_scopes(
    *,
    pages: int,
    total: int,
    debit_count: int,
    credit_count: int,
    debit_total: str,
    credit_total: str,
    period_start: str,
    period_end: str,
    account: str = "1234567890123456",
    holder: str = "测试企业",
) -> list[tuple[int, str]]:
    grand_total = f"{float(debit_total.replace(',', '')) + float(credit_total.replace(',', '')):,.2f}"
    header = "序号 交易日期 交易时间 借方 贷方 余额 摘要 对方账号 对方名称 附言"
    scope = "\n".join(
        (
            "活期账户明细查询",
            f"开始日期：{period_start}",
            f"结束日期：{period_end}",
            f"户名：{holder}",
            f"账号：{account}",
            "币种：人民币",
            f"总笔数：{total}",
            f"借方总金额：{debit_total}",
            f"借方总笔数：{debit_count}",
            f"总金额：{grand_total}",
            f"贷方总金额：{credit_total}",
            f"贷方总笔数：{credit_count}",
            header,
        )
    )
    return [(page, scope) for page in range(1, pages + 1)]


def test_repeated_current_account_query_header_totals_cover_layout_pair() -> None:
    first = _repeated_current_account_query_scopes(
        pages=5,
        total=90,
        debit_count=43,
        credit_count=47,
        debit_total="3,019,670.00",
        credit_total="2,973,220.32",
        period_start="2023-02-23",
        period_end="2023-05-22",
    )
    sibling = _repeated_current_account_query_scopes(
        pages=4,
        total=71,
        debit_count=29,
        credit_count=42,
        debit_total="945,202.30",
        credit_total="977,176.28",
        period_start="2023-05-24",
        period_end="2023-08-24",
    )

    assert resolve_row_count_evidence("", page_texts=first) == RowCountEvidence(90, "header_total", 0.98, 1)
    assert resolve_row_count_evidence("", page_texts=sibling) == RowCountEvidence(71, "header_total", 0.98, 1)


def test_repeated_current_account_query_header_total_fails_closed_on_mutations() -> None:
    valid = _repeated_current_account_query_scopes(
        pages=5,
        total=90,
        debit_count=43,
        credit_count=47,
        debit_total="3,019,670.00",
        credit_total="2,973,220.32",
        period_start="2023-02-23",
        period_end="2023-05-22",
    )

    mutations = []
    missing_scope = valid.copy()
    del missing_scope[2]
    mutations.append(missing_scope)

    changed_count = valid.copy()
    changed_count[1] = (2, changed_count[1][1].replace("借方总笔数：43", "借方总笔数：42"))
    mutations.append(changed_count)

    changed_amount = valid.copy()
    changed_amount[2] = (3, changed_amount[2][1].replace("贷方总金额：2,973,220.32", "贷方总金额：1.00"))
    mutations.append(changed_amount)

    changed_period = valid.copy()
    changed_period[3] = (4, changed_period[3][1].replace("结束日期：2023-05-22", "结束日期：2023-05-23"))
    mutations.append(changed_period)

    reordered_header = valid.copy()
    reordered_header[4] = (
        5,
        reordered_header[4][1].replace("交易日期 交易时间 借方 贷方 余额", "余额 贷方 借方 交易时间 交易日期"),
    )
    mutations.append(reordered_header)

    missing_direction_amount = valid.copy()
    missing_direction_amount[0] = (1, missing_direction_amount[0][1].replace("贷方总金额：2,973,220.32\n", ""))
    mutations.append(missing_direction_amount)

    for scopes in mutations:
        evidence = resolve_row_count_evidence("", page_texts=scopes)
        assert evidence.source not in {
            "split_footer",
            "header_total",
            "statement_header_totals",
            "cumulative_footer_total",
            "page_footer",
        }


def test_repeated_current_account_query_rejects_cross_layout_injection() -> None:
    valid = _repeated_current_account_query_scopes(
        pages=2,
        total=71,
        debit_count=29,
        credit_count=42,
        debit_total="945,202.30",
        credit_total="977,176.28",
        period_start="2023-05-24",
        period_end="2023-08-24",
    )
    injected = valid.copy()
    injected[1] = (
        2,
        injected[1][1]
        .replace("户名：测试企业", "户名：另一企业")
        .replace("账号：1234567890123456", "账号：9999999999999999")
        .replace("活期账户明细查询", "其他银行交易明细"),
    )

    evidence = resolve_row_count_evidence("", page_texts=injected)

    assert evidence.source not in {
        "split_footer",
        "header_total",
        "statement_header_totals",
        "cumulative_footer_total",
        "page_footer",
    }


def test_sparse_page_text_uses_safe_flattened_count_fallback() -> None:
    evidence = resolve_row_count_evidence(
        """
        账户明细 账户名称：测试企业 账号：1234567890123456 起止日期：2025-01-01 - 2025-12-31
        交易日期 交易金额 余额 对方账号 摘要
        交易总金额：380.00 借方累计金额：180.00 贷方累计金额：200.00 总条数: 38
        """,
        page_texts=[(1, "银行流水首页")],
    )

    assert evidence.count == 38
    assert evidence.source == "header_total"
    assert evidence.page is None


def test_page_texts_restore_only_geometry_bound_spdb_header_facts() -> None:
    atoms = [
        {
            "text": text,
            "page_id": "page:0001",
            "source_kind": "pdf_native",
            "confidence": 1.0,
            "bbox": bbox,
        }
        for text, bbox in (
            ("客户名称", [-227.0, 56.0, -183.0, 67.0]),
            ("测试企业", [-177.0, 56.0, -137.0, 67.0]),
            ("账号", [173.0, 76.0, 195.0, 87.0]),
            ("1234567890123456", [203.0, 76.0, 290.0, 87.0]),
            ("账单币种", [-227.0, 96.0, -183.0, 107.0]),
            ("人民币", [-177.0, 96.0, -144.0, 107.0]),
            ("账单统计日期", [173.0, 116.0, 239.0, 127.0]),
            ("2024年01月01日-2024年01月31日", [243.0, 116.0, 402.0, 127.0]),
        )
    ]
    parse_result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, texts=[], tables=[])],
        evidence_plane=SimpleNamespace(evidence={"text_atoms": atoms}),
    )

    assert page_texts_from_parse_result(parse_result) == [
        (
            1,
            "客户名称 测试企业\n账号 1234567890123456\n账单币种 人民币\n"
            "账单统计日期 2024年01月01日-2024年01月31日",
        )
    ]


def test_page_local_borderless_transaction_anchors_are_structural_only() -> None:
    header = "交易日期 交易时间 交易摘要 交易金额 本次余额 对手信息 日志号 交易渠道 交易附言"
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (
                1,
                "\n".join(
                    [
                        header,
                        "20230622 195925 微信支付 -500.00 10705.87 243300133 457272650 电子商务 二维码付款",
                        "20230622 200122 网银在线 -9807.58 898.29 210401293 458045033 电子商务 企业主贷还款",
                    ]
                ),
            ),
            (
                2,
                "\n".join(
                    [
                        header,
                        "20231221 084535 转存 +29000 31953.2 239278739 超级网银 工资",
                    ]
                ),
            ),
        ],
    )

    assert evidence.count == 3
    assert evidence.source == "page_transaction_anchors"
    assert evidence.confidence == 0.80


def test_single_late_page_borderless_header_is_not_a_document_denominator() -> None:
    header = "交易日期 记账日期 摘要 支/收交易金额 账户余额 交易地点 对方户名 对方账户/对方银行"
    continuation = "2022-01-15 2022-01-15 贷款回收 支 -133.80 0.00"
    page_texts = [(page, continuation) for page in range(1, 7)]
    page_texts.append((7, f"{header}\n{continuation}"))

    evidence = resolve_row_count_evidence("", page_texts=page_texts)

    assert evidence.count == 0
    assert evidence.confidence < 0.85


def test_conflicting_page_header_totals_are_not_first_match_document_evidence() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[(1, "总笔数: 10"), (2, "总笔数: 28")],
    )

    assert evidence.count == 0
    assert evidence.source == "none"


def _spdb_segment_page(
    local_page: int,
    declared_pages: int,
    statement_total: int,
    *,
    period: str = "2024年01月01日-2024年01月31日",
) -> str:
    lines = [
        "客户名称 测试企业",
        "账户名称 测试企业",
        "账号 1234567890123456",
        "账单币种 人民币",
        "账单类型 月账单",
        f"账单统计日期 {period}",
        f"第{local_page}页，共{declared_pages}页",
    ]
    if local_page == 1:
        lines.extend(("汇总交易笔数", f"{statement_total}笔", f"汇总交易笔数 {statement_total}笔"))
    return "\n".join(lines)


def _spdb_statement_scopes(segments: list[tuple[int, int]]) -> list[tuple[int, str]]:
    scopes: list[tuple[int, str]] = []
    physical_page = 1
    year, month = 2024, 1
    for declared_pages, statement_total in segments:
        last_day = monthrange(year, month)[1]
        period = f"{year:04d}年{month:02d}月01日-{year:04d}年{month:02d}月{last_day:02d}日"
        for local_page in range(1, declared_pages + 1):
            scopes.append(
                (
                    physical_page,
                    _spdb_segment_page(local_page, declared_pages, statement_total, period=period),
                )
            )
            physical_page += 1
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return scopes


def test_concatenated_complete_statement_segment_totals_are_additive() -> None:
    page_texts = _spdb_statement_scopes([(2, 124), (3, 99), (2, 91)])
    evidence = resolve_row_count_evidence(
        "\n".join(text for _page, text in page_texts),
        page_texts=page_texts,
    )

    assert evidence.count == 314
    assert evidence.source == "statement_header_totals"
    assert evidence.confidence == 0.97


def test_concatenated_first_pages_with_equal_totals_are_both_counted() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=_spdb_statement_scopes([(2, 10), (3, 10)]),
    )

    assert evidence == RowCountEvidence(20, "statement_header_totals", 0.97)


def test_disagreeing_duplicate_header_totals_on_one_page_fail_closed() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[(1, "第1页,共2页\n汇总交易笔数\n10笔\n汇总交易笔数\n11笔")],
    )

    assert evidence == RowCountEvidence.empty()


def test_statement_header_totals_reject_missing_segment_or_partial_segment() -> None:
    complete = _spdb_statement_scopes([(2, 10), (3, 20), (2, 30)])
    missing_segment = complete[:2] + complete[5:]
    missing_segment = [(page, text) for page, (_old_page, text) in enumerate(missing_segment, 1)]
    partial_segment = complete[:-1]

    missing = resolve_row_count_evidence("", page_texts=missing_segment)
    partial = resolve_row_count_evidence("", page_texts=partial_segment)

    assert missing.source != "statement_header_totals"
    assert partial.source != "statement_header_totals"


def test_statement_header_totals_reject_duplicate_or_reordered_source_scope() -> None:
    complete = _spdb_statement_scopes([(2, 10), (2, 20)])
    duplicated = complete.copy()
    duplicated[1] = (2, complete[0][1])
    reordered = complete.copy()
    reordered[0], reordered[1] = (1, complete[1][1]), (2, complete[0][1])

    duplicate_evidence = resolve_row_count_evidence("", page_texts=duplicated)
    reordered_evidence = resolve_row_count_evidence("", page_texts=reordered)

    assert duplicate_evidence.source != "statement_header_totals"
    assert reordered_evidence.source != "statement_header_totals"


def test_statement_header_totals_reject_period_gap_overlap_or_partial_month() -> None:
    complete = _spdb_statement_scopes([(1, 10), (1, 20), (1, 30)])
    gap = complete.copy()
    gap[1] = (2, gap[1][1].replace("2024年02月01日-2024年02月29日", "2024年03月01日-2024年03月31日"))
    overlap = complete.copy()
    overlap[1] = (2, overlap[1][1].replace("2024年02月01日-2024年02月29日", "2024年01月01日-2024年01月31日"))
    partial = complete.copy()
    partial[1] = (2, partial[1][1].replace("2024年02月01日-2024年02月29日", "2024年02月02日-2024年02月29日"))

    for malformed in (gap, overlap, partial):
        evidence = resolve_row_count_evidence("", page_texts=malformed)
        assert evidence.source != "statement_header_totals"


def test_spdb_annual_bilingual_segment_does_not_require_monthly_label() -> None:
    scopes = []
    for page in range(1, 3):
        lines = [
            "客户名称 Customer Name 测试企业",
            "账户名称 Account Name 测试企业",
            "账号 Account Number 1234567890123456",
            "账单币种 Currency 人民币 CNY",
            "账单统计日期 Start Time & End Time 2025/01/01 - 2025/12/31",
            f"第{page}页，共2页",
        ]
        if page == 1:
            lines.append("汇总交易笔数 17笔")
        scopes.append((page, "\n".join(lines)))

    assert resolve_row_count_evidence("", page_texts=scopes) == RowCountEvidence(17, "header_total", 0.94, 1)


def test_statement_header_totals_reject_unrelated_page_ids_or_cross_layout_injection() -> None:
    unrelated_page_ids = [
        (7, _spdb_segment_page(1, 1, 10)),
        (8, _spdb_segment_page(1, 1, 20)),
    ]
    cross_layout = [
        (1, "其他银行账户明细\n第1页，共1页\n汇总交易笔数 10笔"),
        (2, "其他银行账户明细\n第1页，共1页\n汇总交易笔数 20笔"),
    ]

    unrelated = resolve_row_count_evidence("", page_texts=unrelated_page_ids)
    injected = resolve_row_count_evidence("", page_texts=cross_layout)

    assert unrelated.source != "statement_header_totals"
    assert injected.source != "statement_header_totals"


def test_page_local_record_count_label_is_bounded_count_evidence() -> None:
    page = """
    用户所属公司: 测试企业
    打印时间: 2026/02/24 16:19
    记录数: 71
    交易日期
    借方(出账)
    贷方(入账)
    余额摘要
    收(付)方名称
    收(付)方账号
    交易类型
    """
    evidence = resolve_row_count_evidence("", page_texts=[(1, page)])

    assert evidence.count == 71


def test_sparse_spdb_summary_cannot_establish_document_completeness() -> None:
    text = """
    期末余额
    Ending Balance\t24,175.16\t汇总交易笔数
    Total number of transactions\t174笔\t汇总借方发生额
    The Total Debit Amount\t1,047,323.06\t汇总贷方发生额
    The Total Credit Amount\t1,063,069.78
    """

    evidence = resolve_row_count_evidence(text, page_texts=[(1, "rotated sparse page text")])

    assert evidence.count == 0
    assert evidence.source == "none"


def test_column_major_direction_summary_keeps_counts_distinct_from_money_totals() -> None:
    text = """
    交易明细
    账号：1234567890123456
    企业名称：测试企业
    数据时间范围：2024年01月01日-2024年01月31日
    支出总笔数：
    28
    支出总金额：
    收入总笔数：
    20592403.75
    6
    收入总金额：
    18282600.57
    交易日期 交易时间 收支标志 交易金额 对方户名 账户余额
    """
    records = [
        *[{"normalized": {"date": "2024-01-01", "direction": "expense", "amount": 735442.991071}} for _ in range(28)],
        *[{"normalized": {"date": "2024-01-01", "direction": "income", "amount": 3047100.095}} for _ in range(6)],
    ]

    evidence = resolve_row_count_evidence("", page_texts=[(4, text)])
    failures = audit_bank_statement_invariants(records, "", page_texts=[(4, text)])

    assert evidence.count == 34
    assert evidence.source == "split_footer"
    assert not any("count" in failure or "total" in failure for failure in failures)


def _signed_reversal_column_major_summary(debit_total: str) -> str:
    return f"""
    交易明细
    账号：1234567890123456
    企业名称：测试企业
    数据时间范围：2024年01月01日-2024年01月31日
    支出总笔数：
    2
    支出总金额：
    收入总笔数：
    {debit_total}
    1
    收入总金额：
    5.00
    交易日期 交易时间 收支标志 交易金额 对方户名 账户余额
    """


def _signed_reversal_invariant_records() -> list[dict]:
    return [
        {
            "raw": {"借/贷": "借方", "交易金额": "10.00"},
            "normalized": {"date": "2024-01-01", "direction": "expense", "amount": 10.0},
            "canonical_raw": {"direction": "借方", "amount": "10.00"},
        },
        {
            "raw": {"借/贷": "借方", "交易金额": "-3.00"},
            "normalized": {"date": "2024-01-02", "direction": "expense", "amount": 3.0},
            "canonical_raw": {"direction": "借方", "amount": "-3.00"},
        },
        {
            "raw": {"借/贷": "贷方", "交易金额": "5.00"},
            "normalized": {"date": "2024-01-03", "direction": "income", "amount": 5.0},
            "canonical_raw": {"direction": "贷方", "amount": "5.00"},
        },
    ]


def test_invariant_totals_accept_complete_source_signed_reversal_plane() -> None:
    text = _signed_reversal_column_major_summary("7.00")
    records = _signed_reversal_invariant_records()

    evidence = resolve_row_count_evidence("", page_texts=[(1, text)])
    failures = audit_bank_statement_invariants(records, "", page_texts=[(1, text)])

    assert evidence == RowCountEvidence(3, "split_footer", 0.98, 1)
    assert failures == []
    assert records[1]["normalized"] == {
        "date": "2024-01-02",
        "direction": "expense",
        "amount": 3.0,
    }


def test_invariant_signed_reversal_plane_fails_closed_on_provenance_gaps() -> None:
    amount_mismatch = _signed_reversal_invariant_records()
    amount_mismatch[1]["raw"]["交易金额"] = "-2.00"
    amount_mismatch[1]["canonical_raw"]["amount"] = "-2.00"
    sign_only = _signed_reversal_invariant_records()
    sign_only[1]["raw"].pop("借/贷")
    unowned_positive = _signed_reversal_invariant_records()
    contaminated_direction = "借方，以网点对账单为准。客服电话：95595"
    unowned_positive[0]["raw"]["借/贷"] = contaminated_direction
    unowned_positive[0]["canonical_raw"]["direction"] = contaminated_direction

    amount_failures = audit_bank_statement_invariants(
        amount_mismatch,
        "",
        page_texts=[(1, _signed_reversal_column_major_summary("8.00"))],
    )
    sign_only_failures = audit_bank_statement_invariants(
        sign_only,
        "",
        page_texts=[(1, _signed_reversal_column_major_summary("7.00"))],
    )
    unowned_positive_failures = audit_bank_statement_invariants(
        unowned_positive,
        "",
        page_texts=[(1, _signed_reversal_column_major_summary("7.00"))],
    )

    assert "bank_invariant_failed:debit_total:13.00/8.00" in amount_failures
    assert "bank_invariant_failed:debit_total:13.00/7.00" in sign_only_failures
    assert "bank_invariant_failed:debit_total:13.00/7.00" in unowned_positive_failures


def test_row_level_directions_outrank_contradictory_footer_counts_when_balance_chain_closes() -> None:
    records = [
        {
            "normalized": {"date": "2024-01-01", "direction": "income", "amount": 100.0, "balance": 100.0},
            "canonical_raw": {"direction": "贷方"},
        },
        {
            "normalized": {"date": "2024-01-02", "direction": "expense", "amount": 20.0, "balance": 80.0},
            "canonical_raw": {"direction": "借方"},
        },
        {
            "normalized": {"date": "2024-01-03", "direction": "income", "amount": 5.0, "balance": 85.0},
            "canonical_raw": {"direction": "贷方"},
        },
        {
            "normalized": {"date": "2024-01-04", "direction": "expense", "amount": 10.0, "balance": 75.0},
            "canonical_raw": {"direction": "借方"},
        },
    ]
    text = "借方笔数：3 贷方笔数：1"

    failures = audit_bank_statement_invariants(records, text)

    assert not any("debit_count" in failure or "credit_count" in failure for failure in failures)


def test_authoritative_selected_census_outranks_weaker_text_anchor_in_invariant_audit() -> None:
    records = [
        {"normalized": {"date": f"2024-01-{(index % 28) + 1:02d}", "direction": "income", "amount": 1.0}}
        for index in range(150)
    ]
    page_texts = [(page, "交易日期 交易金额 账户余额\n20240101 1.00 1.00") for page in range(1, 11)]

    failures = audit_bank_statement_invariants(
        records,
        "",
        page_texts=page_texts,
        row_count_evidence=RowCountEvidence(150, "ocr_page_ordinal_census", 0.99),
    )

    assert "bank_invariant_failed:row_count:150/10" not in failures


def test_low_confidence_candidate_count_cannot_override_text_anchor_in_invariant_audit() -> None:
    records = [
        {"normalized": {"date": f"2024-01-{index:02d}", "direction": "income", "amount": 1.0}} for index in range(1, 4)
    ]
    header = "交易日期 交易时间 交易摘要 交易金额 本次余额 对手信息 日志号 交易渠道 交易附言"
    page_texts = [(1, f"{header}\n20240101 090000 付款 -1.00 9.00 123456 789 柜面 货款")]

    failures = audit_bank_statement_invariants(
        records,
        "",
        page_texts=page_texts,
        row_count_evidence=RowCountEvidence(3, "candidate_rows", 0.55),
    )

    assert not any(item.startswith("bank_invariant_failed:row_count") for item in failures)


def test_repeated_identical_document_total_is_deduplicated() -> None:
    contract = """
    账户明细 账户名称：测试企业 账号：1234567890123456 起止日期：2025-01-01 - 2025-12-31
    交易日期 交易金额 余额 对方账号 摘要
    """
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, f"{contract}\n总笔数: 549"),
            (2, f"{contract}\n总笔数: 549"),
            (3, f"{contract}\n总笔数: 549"),
        ],
    )

    assert evidence.count == 549
    assert evidence.source == "header_total"


def test_page_local_borderless_count_accepts_stacked_first_row_cells() -> None:
    header = "记账日期 货币 交易金额 余额 交易摘要 对手信息"
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (
                1,
                "\n".join(
                    [
                        header,
                        "2024-01-01",
                        "CNY",
                        "0.07",
                        "0.07",
                        "结息",
                    ]
                ),
            ),
            (
                2,
                "\n".join(
                    [
                        header,
                        "2024-01-02 CNY -1.00 9.00 付款 测试有限公司 123456789012345",
                    ]
                ),
            ),
        ],
    )

    assert evidence.count == 2
    assert evidence.source == "page_transaction_anchors"


def test_borderless_anchor_count_excludes_query_period_dates_before_table_header() -> None:
    header = "交易日期 交易时间 交易摘要 交易金额 本次余额 对手信息"
    page_text = "\n".join(
        [
            "起止日期",
            "2022-07-01",
            "0",
            "0",
            "2023-04-15",
            "0",
            "0",
            header,
            "2023-04-14 120000 转存 +1134.30 1168.10 应付商户延迟清算款",
            "2023-04-15 120000 转存 +696.50 1864.60 应付商户延迟清算款",
        ]
    )

    evidence = resolve_row_count_evidence("", page_texts=[(1, page_text)])

    assert evidence.count == 2
    assert evidence.source == "page_transaction_anchors"


def _ccb_primary_page(start: int, end: int, *, corrupt_date: str = "") -> str:
    rows: list[str] = []
    for sequence in range(start, end + 1):
        rows.extend(
            [
                str(sequence),
                "银联入账",
                "人民币元",
                "钞",
                corrupt_date if sequence == end and corrupt_date else "20230803",
            ]
        )
    return "\n".join(
        [
            "卡号/账号:6217001180023847257",
            "客户名称：测试客户",
            "序号",
            "摘要",
            "币别",
            "钞汇",
            "交易日期",
            *rows,
            "生成时间:2023-09-11 14:51:30",
            "序号 摘要 币别 钞汇 交易日期 交易金额 账户余额",
            *(str(sequence) for sequence in range(start, end + 1)),
        ]
    )


def test_ccb_primary_source_sequence_census_proves_non_one_span() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, _ccb_primary_page(571, 589)),
            (2, _ccb_primary_page(590, 608)),
            (3, _ccb_primary_page(609, 617)),
        ],
    )

    assert evidence == RowCountEvidence(47, "ccb_primary_source_sequence", 0.80)


def test_ccb_primary_source_sequence_rejects_gap_duplicate_plane_and_invalid_tuple() -> None:
    gap = resolve_row_count_evidence(
        "",
        page_texts=[(1, _ccb_primary_page(571, 589)), (2, _ccb_primary_page(591, 609))],
    )
    invalid = resolve_row_count_evidence(
        "",
        page_texts=[(1, _ccb_primary_page(571, 589, corrupt_date="20230230"))],
    )
    complete = _ccb_primary_page(571, 589)
    lines = complete.splitlines()
    footer_index = next(index for index, line in enumerate(lines) if line.startswith("生成时间:"))
    terminal_primary_tuple_deleted = "\n".join([*lines[: footer_index - 5], *lines[footer_index:]])
    terminal_duplicate_ordinal_deleted = "\n".join(lines[:-1])
    terminal_missing = resolve_row_count_evidence(
        "",
        page_texts=[(1, terminal_primary_tuple_deleted)],
    )
    duplicate_terminal_missing = resolve_row_count_evidence(
        "",
        page_texts=[(1, terminal_duplicate_ordinal_deleted)],
    )

    assert gap.source != "ccb_primary_source_sequence"
    assert invalid.source != "ccb_primary_source_sequence"
    assert terminal_missing.source != "ccb_primary_source_sequence"
    assert duplicate_terminal_missing.source != "ccb_primary_source_sequence"


def test_ccb_primary_source_sequence_rejects_duplicate_plane_internal_gap() -> None:
    complete = _ccb_primary_page(571, 589)
    duplicate_gap = complete.replace("\n580\n581\n582\n", "\n580\n582\n", 1)

    evidence = resolve_row_count_evidence("", page_texts=[(1, duplicate_gap)])

    assert evidence.source != "ccb_primary_source_sequence"


def test_ccb_primary_source_sequence_rejects_cross_layout_header_order() -> None:
    complete = _ccb_primary_page(571, 589)
    reordered_duplicate_header = complete.replace(
        "序号 摘要 币别 钞汇 交易日期 交易金额 账户余额",
        "序号 摘要 币别 钞汇 交易日期 账户余额 交易金额",
        1,
    )

    evidence = resolve_row_count_evidence("", page_texts=[(1, reordered_duplicate_header)])

    assert evidence.source != "ccb_primary_source_sequence"


def _cmb_primary_page(page: int, total: int, rows: list[tuple[str, str, str, str]]) -> str:
    source_rows: list[str] = []
    for transaction_date, currency, amount, balance in rows:
        source_rows.extend([transaction_date, currency, amount, balance, "转账汇款", "测试对手"])
    prefix = (
        [
            "招商银行交易流水",
            "Transaction Statement of China Merchants Bank",
            "2022-03-28 -- 2023-03-28",
            "申请时间：2023-03-30 09:22:28",
        ]
        if page == 1
        else []
    )
    return "\n".join(
        [
            *prefix,
            "记账日期",
            "货币",
            "交易金额",
            "联机余额",
            "交易摘要",
            "对手信息",
            "Date",
            "Currency",
            "Transaction",
            "Amount",
            "Balance",
            "Transaction Type",
            "Counter Party",
            *source_rows,
            f"{page}/{total}",
            "Transaction Statement of China Merchants Bank",
            "记账日期 货币 交易金额 联机余额 交易摘要 对手信息",
            "Date Currency Transaction Amount Balance Transaction Type Counter Party",
            *(" ".join(row) + " 转账汇款 测试对手" for row in rows),
        ]
    )


def test_cmb_bilingual_primary_source_census_counts_continuation_page_once() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (
                1,
                _cmb_primary_page(
                    1,
                    2,
                    [
                        ("2022-10-15", "CNY", "56.31", "56.31"),
                        ("2022-10-17", "CNY", "-56.31", "0.00"),
                    ],
                ),
            ),
            (2, _cmb_primary_page(2, 2, [("2023-03-15", "CNY", "-5,451.60", "0.00")])),
        ],
    )

    assert evidence == RowCountEvidence(3, "cmb_primary_source_rows", 0.80)


def test_cmb_primary_source_census_counts_mixed_currencies_and_signed_amounts() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (
                1,
                _cmb_primary_page(
                    1,
                    1,
                    [
                        ("2022-10-15", "CNY", "+56.31", "56.31"),
                        ("2022-10-17", "USD", "-7.00", "49.31"),
                    ],
                ),
            )
        ],
    )

    assert evidence == RowCountEvidence(2, "cmb_primary_source_rows", 0.80)


def test_cmb_primary_source_census_fails_closed_on_unsupported_dated_row() -> None:
    page = _cmb_primary_page(1, 1, [("2022-10-15", "US D", "56.31", "56.31")])

    evidence = resolve_row_count_evidence("", page_texts=[(1, page)])

    assert evidence.source != "cmb_primary_source_rows"


def test_cmb_primary_source_census_rejects_unaccounted_date_shaped_tuple() -> None:
    complete = _cmb_primary_page(
        1,
        1,
        [
            ("2022-10-15", "CNY", "56.31", "56.31"),
            ("2022-10-17", "CNY", "-56.31", "0.00"),
        ],
    )

    for unsupported_date in ("2022/10/17", "20221017"):
        primary_tuple_unrecognized = complete.replace(
            "2022-10-17\nCNY\n-56.31\n0.00",
            f"{unsupported_date}\nCNY\n-56.31\n0.00",
            1,
        )
        evidence = resolve_row_count_evidence("", page_texts=[(1, primary_tuple_unrecognized)])

        assert evidence.source != "cmb_primary_source_rows"


def test_cmb_primary_source_census_rejects_missing_header_or_wrong_page_marker() -> None:
    valid = _cmb_primary_page(1, 2, [("2022-10-15", "CNY", "56.31", "56.31")])
    missing_header = _cmb_primary_page(2, 2, [("2023-03-15", "CNY", "-5,451.60", "0.00")]).replace(
        "Counter Party\n",
        "",
        1,
    )
    wrong_marker = _cmb_primary_page(2, 3, [("2023-03-15", "CNY", "-5,451.60", "0.00")])

    missing = resolve_row_count_evidence("", page_texts=[(1, valid), (2, missing_header)])
    mismatched = resolve_row_count_evidence("", page_texts=[(1, valid), (2, wrong_marker)])

    assert missing.source != "cmb_primary_source_rows"
    assert mismatched.source != "cmb_primary_source_rows"


def test_cmb_primary_source_census_rejects_cross_layout_header_order() -> None:
    complete = _cmb_primary_page(1, 1, [("2022-10-15", "CNY", "56.31", "56.31")])
    reordered_source_header = complete.replace(
        "Date\nCurrency\nTransaction\nAmount\nBalance",
        "Date\nTransaction\nCurrency\nAmount\nBalance",
        1,
    )

    evidence = resolve_row_count_evidence("", page_texts=[(1, reordered_source_header)])

    assert evidence.source != "cmb_primary_source_rows"


def test_cmb_primary_source_census_rejects_asymmetric_terminal_row_loss() -> None:
    rows = [
        ("2022-10-15", "CNY", "56.31", "56.31"),
        ("2022-10-17", "USD", "-7.00", "49.31"),
        ("2022-10-19", "EUR", "+1.00", "50.31"),
    ]
    complete = _cmb_primary_page(1, 1, rows)
    primary_terminal = "\n".join((*rows[-1], "转账汇款", "测试对手"))
    duplicate_terminal = " ".join(rows[-1]) + " 转账汇款 测试对手"
    primary_terminal_missing = complete.replace(f"\n{primary_terminal}\n1/1", "\n1/1", 1)
    duplicate_terminal_missing = complete.removesuffix(f"\n{duplicate_terminal}")

    primary_evidence = resolve_row_count_evidence("", page_texts=[(1, primary_terminal_missing)])
    duplicate_evidence = resolve_row_count_evidence("", page_texts=[(1, duplicate_terminal_missing)])

    assert primary_evidence.source != "cmb_primary_source_rows"
    assert duplicate_evidence.source != "cmb_primary_source_rows"


def test_cmb_symmetric_row_plane_loss_remains_non_authoritative() -> None:
    rows = [
        ("2022-10-15", "CNY", "56.31", "56.31"),
        ("2022-10-17", "USD", "-7.00", "49.31"),
        ("2022-10-19", "EUR", "+1.00", "50.31"),
    ]
    shortened = _cmb_primary_page(1, 1, rows[:-1])

    evidence = resolve_row_count_evidence("", page_texts=[(1, shortened)])

    assert evidence == RowCountEvidence(2, "cmb_primary_source_rows", 0.80)


def test_ccb_symmetric_row_plane_loss_remains_non_authoritative() -> None:
    evidence = resolve_row_count_evidence("", page_texts=[(1, _ccb_primary_page(571, 588))])

    assert evidence == RowCountEvidence(18, "ccb_primary_source_sequence", 0.80)


def test_cmb_primary_source_census_rejects_duplicate_internal_gap_or_extra_row() -> None:
    rows = [
        ("2022-10-15", "CNY", "56.31", "56.31"),
        ("2022-10-17", "USD", "-7.00", "49.31"),
        ("2022-10-19", "EUR", "+1.00", "50.31"),
    ]
    complete = _cmb_primary_page(1, 1, rows)
    duplicate_middle = " ".join(rows[1]) + " 转账汇款 测试对手"
    duplicate_gap = complete.replace(f"\n{duplicate_middle}\n", "\n", 1)
    duplicate_extra = complete + "\n2022-10-20 GBP 2.00 52.31 转账汇款 测试对手"
    first_duplicate = " ".join(rows[0]) + " 转账汇款 测试对手"
    second_duplicate = " ".join(rows[1]) + " 转账汇款 测试对手"
    duplicate_reordered = complete.replace(
        f"\n{first_duplicate}\n{second_duplicate}\n",
        f"\n{second_duplicate}\n{first_duplicate}\n",
        1,
    )

    gap_evidence = resolve_row_count_evidence("", page_texts=[(1, duplicate_gap)])
    extra_evidence = resolve_row_count_evidence("", page_texts=[(1, duplicate_extra)])
    reordered_evidence = resolve_row_count_evidence("", page_texts=[(1, duplicate_reordered)])

    assert gap_evidence.source != "cmb_primary_source_rows"
    assert extra_evidence.source != "cmb_primary_source_rows"
    assert reordered_evidence.source != "cmb_primary_source_rows"


def test_cmb_primary_source_census_requires_exact_repeated_plane_signature() -> None:
    complete = _cmb_primary_page(1, 1, [("2022-10-15", "CNY", "56.31", "56.31")])
    duplicate_title_and_header = (
        "Transaction Statement of China Merchants Bank\n"
        "记账日期 货币 交易金额 联机余额 交易摘要 对手信息"
    )
    missing_title = complete.replace(
        duplicate_title_and_header,
        "记账日期 货币 交易金额 联机余额 交易摘要 对手信息",
        1,
    )
    missing_header = complete.replace(
        duplicate_title_and_header,
        "Transaction Statement of China Merchants Bank",
        1,
    )
    reordered_header = complete.replace(
        "记账日期 货币 交易金额 联机余额 交易摘要 对手信息",
        "记账日期 货币 联机余额 交易金额 交易摘要 对手信息",
        1,
    )

    for malformed in (missing_title, missing_header, reordered_header):
        evidence = resolve_row_count_evidence("", page_texts=[(1, malformed)])
        assert evidence.source != "cmb_primary_source_rows"


def test_cmb_marker_final_continuation_page_uses_its_bounded_primary_plane() -> None:
    first = _cmb_primary_page(1, 2, [("2022-10-15", "CNY", "56.31", "56.31")])
    continuation = _cmb_primary_page(2, 2, [("2022-10-17", "CNY", "-56.31", "0.00")])
    marker_final_continuation = continuation.split("\n2/2", 1)[0] + "\n2/2"

    evidence = resolve_row_count_evidence(
        "",
        page_texts=[(1, first), (2, marker_final_continuation)],
    )

    # The continuation has only one serialized source plane.  The exact
    # header-to-marker scope defines its source truth; it is not evidence that
    # a separately emitted candidate is complete.
    assert evidence == RowCountEvidence(2, "cmb_primary_source_rows", 0.80)


def test_page_footer_transaction_counts_are_summed_across_pages() -> None:
    def page(page: int, count: int) -> tuple[int, str]:
        return page, "\n".join(
            (
                "个人手机银行交易明细",
                "户名：测试用户",
                "卡号/账号：1234567890123456",
                "起始日期：2024-01-01 终止日期：2024-12-31",
                f"第{page}页 共4页 本页支出合计: 10.00 本页收入合计: 20.00 本页交易笔数: {count}",
                "交易日期 对方户名 对方账号 交易摘要 发生额 余额 币种",
            )
        )

    assert resolve_row_count_evidence("", page_texts=[page(1, 28), page(2, 28), page(3, 28), page(4, 25)]).count == 109


def test_page_footer_count_requires_complete_unique_page_scope() -> None:
    missing = "\n".join(
        (
            "第1页 共3页 本页交易笔数: 10",
            "第3页 共3页 本页交易笔数: 8",
        )
    )
    duplicated = "\n".join(
        (
            "第1页 共2页 本页交易笔数: 10",
            "第1页 共2页 本页交易笔数: 10",
        )
    )

    assert resolve_row_count_evidence(missing).source != "page_footer"
    assert resolve_row_count_evidence(duplicated).source != "page_footer"


def test_isolated_generic_count_phrase_is_not_issuer_authority() -> None:
    evidence = resolve_row_count_evidence("任意说明\n记录数: 5")

    assert evidence == RowCountEvidence.empty()


def _contracted_split_count_text(header: str) -> str:
    return "\n".join(
        (
            "账户明细",
            "账户名称：测试企业",
            "账号：1234567890123456",
            "起止日期：2025-01-01 - 2025-12-31",
            header,
            "借方笔数: 2 借方发生总额: 10.00 贷方笔数: 3 贷方发生总额: 20.00",
        )
    )


def test_issuer_count_requires_a_bounded_ordered_ledger_header() -> None:
    accepted = resolve_row_count_evidence(
        _contracted_split_count_text("交易日期 交易金额 账户余额 对方户名")
    )
    rejected_headers = (
        "对方户名 账户余额 交易金额 交易日期",
        "账户余额 交易金额 交易日期 对方户名",
        "交易日期、交易金额、账户余额、对方户名如下",
        "交易日期\n甲乙丙丁\n交易金额\n账户余额\n对方户名",
    )

    assert accepted == RowCountEvidence(5, "split_footer", 0.98)
    for header in rejected_headers:
        assert resolve_row_count_evidence(_contracted_split_count_text(header)) == RowCountEvidence.empty()


def test_issuer_count_header_roles_cannot_be_scattered_across_page_scopes() -> None:
    page_one = "\n".join(
        (
            "账户明细",
            "账户名称：测试企业",
            "账号：1234567890123456",
            "起止日期：2025-01-01 - 2025-12-31",
            "交易日期 交易金额",
        )
    )
    page_two = "\n".join(
        (
            "账户余额 对方户名",
            "借方笔数: 2 借方发生总额: 10.00 贷方笔数: 3 贷方发生总额: 20.00",
        )
    )

    assert resolve_row_count_evidence("", page_texts=[(1, page_one), (2, page_two)]) == RowCountEvidence.empty()


def _cumulative_scope(
    page: int,
    total_pages: int,
    through: int,
    total_rows: int,
    *,
    account: str = "1234567890123456",
    holder: str = "测试企业",
) -> str:
    return "\n".join(
        (
            "单位账户明细对账单",
            "20240101-20240131",
            "账户名称:",
            holder,
            "客户账号:",
            account,
            "开户机构:",
            "测试分行营业部",
            "币种:",
            "人民币",
            "交易日期",
            "交易金额",
            "账户余额",
            "对方户名",
            "对方账号",
            "摘要/备注",
            "编号",
            f"第{page}页，共{total_pages}页/第{through}笔，共{total_rows}笔",
        )
    )


def test_cumulative_transaction_footer_proves_complete_document_count() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, _cumulative_scope(1, 3, 36, 73)),
            (2, _cumulative_scope(2, 3, 72, 73)),
            (3, _cumulative_scope(3, 3, 73, 73)),
        ],
    )

    assert evidence == RowCountEvidence(73, "cumulative_footer_total", 0.99, 3)


def test_cumulative_transaction_footer_fails_closed_on_conflict_or_incomplete_final() -> None:
    conflicting = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, _cumulative_scope(1, 2, 36, 73)),
            (2, _cumulative_scope(2, 2, 72, 74)),
        ],
    )
    incomplete = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, _cumulative_scope(1, 2, 36, 73)),
            (2, _cumulative_scope(2, 2, 72, 73)),
        ],
    )

    assert conflicting.count == 0
    assert incomplete.count == 0


def test_cumulative_transaction_footer_rejects_missing_page_scope() -> None:
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, "第36笔，共73笔"),
            (3, "第73笔，共73笔"),
        ],
    )

    assert evidence.count == 0


def test_cumulative_footer_requires_ordered_header_and_stable_identity() -> None:
    mismatched_identity = resolve_row_count_evidence(
        "",
        page_texts=[
            (1, _cumulative_scope(1, 2, 10, 20)),
            (2, _cumulative_scope(2, 2, 20, 20, holder="另一企业")),
        ],
    )
    reordered = _cumulative_scope(1, 1, 20, 20).replace(
        "交易金额\n账户余额",
        "账户余额\n交易金额",
        1,
    )

    assert mismatched_identity.source != "cumulative_footer_total"
    assert resolve_row_count_evidence("", page_texts=[(1, reordered)]).source != "cumulative_footer_total"


def test_cumulative_footer_reconciles_compact_duplicate_identity_plane() -> None:
    page_one = _cumulative_scope(1, 2, 10, 20)
    page_two = _cumulative_scope(2, 2, 20, 20)
    compact_identity = (
        "账户名称: 测试企业 客户账号: 1234567890123456 "
        "开户机构: 测试分行营业部 币种: 人民币"
    )
    evidence = resolve_row_count_evidence(
        "",
        page_texts=[(1, page_one), (2, f"{page_two}\n{compact_identity}\n{compact_identity}")],
    )

    assert evidence == RowCountEvidence(20, "cumulative_footer_total", 0.99, 2)
    for conflicting_identity in (
        "账户名称: 另一企业 客户账号: 1234567890123456 开户机构: 测试分行营业部 币种: 人民币",
        "账户名称: 测试企业 客户账号: 9999999999999999 开户机构: 测试分行营业部 币种: 人民币",
        "账户名称: 测试企业 客户账号: 1234567890123456 开户机构: 另一分行 币种: 人民币",
        "账户名称: 测试企业 客户账号: 1234567890123456 开户机构: 测试分行营业部 币种: CNY",
    ):
        conflicting = resolve_row_count_evidence(
            "",
            page_texts=[(1, page_one), (2, f"{page_two}\n{conflicting_identity}")],
        )
        assert conflicting.source != "cumulative_footer_total"


def test_corporate_detail_header_identity_and_split_footer_counts_are_bounded() -> None:
    text = """
    对公客户账户明细
    账    号: 5106010120010001125
    币    种: 人民币
    客户名称: 重庆正大华日软件有限公司
    开户机构: 510601
    起始日期: 20250101
    终止日期: 20251231
    交易日期 交易发生金额 账户余额 对方账号 对方户名 摘要 备注
    20250121 -15069.44 1993153.47 5101010179730017689 重庆正大华日软件有限公司 归还本息
    贷款账号: 5101010179730017689
    借方合计笔数：65笔
    借方合计金额：23445508.32元
    贷方合计笔数：12笔
    贷方合计金额：20020834.89元
    """.strip()

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "重庆正大华日软件有限公司"
    assert identity["account_number"] == "5106010120010001125"
    assert count_expected_rows_from_bank_footer(text) == 77


def test_reverse_order_balance_chain_passes_when_totals_close() -> None:
    text = "收入总金额： 100.00 收入总笔数： 1 支出总金额： 20.00 支出总笔数： 1"
    records = [
        {"normalized": {"date": "2023-01-02", "amount": 100.0, "direction": "income", "balance": 180.0}},
        {"normalized": {"date": "2023-01-01", "amount": 20.0, "direction": "expense", "balance": 80.0}},
    ]

    assert audit_bank_statement_invariants(records, text) == []


def test_balance_chain_skips_inside_and_across_identical_explicit_timestamp_batch() -> None:
    records = [
        {
            "normalized": {
                "sequence_no": "1",
                "date": "2023-06-30",
                "timestamp": "2023-06-30T09:59:59",
                "amount": 100.0,
                "direction": "income",
                "balance": 100.0,
            }
        },
        {
            "normalized": {
                "sequence_no": "2",
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:00:00",
                "amount": 60.0,
                "direction": "income",
                "balance": 140.0,
            }
        },
        {
            "normalized": {
                "sequence_no": "3",
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:00:00",
                "amount": 40.0,
                "direction": "income",
                "balance": 160.0,
            }
        },
        {
            "normalized": {
                "sequence_no": "4",
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:01:00",
                "amount": 10.0,
                "direction": "expense",
                "balance": 150.0,
            }
        },
        {
            "normalized": {
                "sequence_no": "5",
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:02:00",
                "amount": 10.0,
                "direction": "expense",
                "balance": 140.0,
            }
        },
    ]

    failures = audit_bank_statement_invariants(records, "")

    assert not any(item.startswith("bank_invariant_failed:balance_chain") for item in failures)
    assert not any(item.startswith("bank_review:balance_chain_gap") for item in failures)


def test_balance_chain_still_validates_unique_explicit_timestamps() -> None:
    records = [
        {
            "normalized": {
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:00:00",
                "amount": 100.0,
                "direction": "income",
                "balance": 100.0,
            }
        },
        {
            "normalized": {
                "date": "2023-06-30",
                "timestamp": "2023-06-30T10:01:00",
                "amount": 10.0,
                "direction": "expense",
                "balance": 50.0,
            }
        },
    ]

    failures = audit_bank_statement_invariants(records, "")

    assert "bank_invariant_failed:balance_chain:1/1" in failures


def test_reconciliation_direction_counts_are_checked_independently() -> None:
    text = """
    账户对账单 账户名称：测试企业 账号：1234567890123456 起止日期：2023-01-01 - 2023-01-31
    交易日期 交易金额 余额 对方账号 摘要
    借方笔数：2 借方发生总额：20.00 贷方笔数：1 贷方发生总额：100.00 合计笔数：3
    """
    records = [
        {"normalized": {"date": "2023-01-01", "amount": 20.0, "direction": "expense"}},
        {"normalized": {"date": "2023-01-02", "amount": 100.0, "direction": "income"}},
    ]

    failures = audit_bank_statement_invariants(records, text)

    assert "bank_invariant_failed:row_count:2/3" in failures
    assert "bank_invariant_failed:debit_count:1/2" in failures
    assert "bank_invariant_failed:credit_count:1/1" not in failures


def test_emitted_record_sequence_cannot_override_independent_page_anchor_count() -> None:
    records = [
        {
            "raw": {"序号": str(index)},
            "normalized": {
                "sequence_no": str(index),
                "date": f"2024-01-0{index}",
                "direction": "income",
                "amount": 1.0,
            },
        }
        for index in range(1, 5)
    ]

    failures = audit_bank_statement_invariants(
        records,
        "",
        row_count_evidence=RowCountEvidence(5, "page_transaction_anchors", 0.99),
    )

    assert not any(item.startswith("bank_invariant_failed:row_count") for item in failures)


def test_balance_chain_gap_reports_review_only_missing_row_candidate() -> None:
    records = [
        {"normalized": {"date": "2022-06-24", "amount": 2.0, "direction": "expense", "balance": 89.60}},
        {"normalized": {"date": "2022-07-01", "amount": 16.99, "direction": "expense", "balance": 44.82}},
    ]

    failures = audit_bank_statement_invariants(records, "")

    assert "bank_invariant_failed:balance_chain:1/1" in failures
    assert (
        "bank_review:balance_chain_gap:"
        "row=2:date=2022-07-01:direction=expense:amount=16.99:"
        "prev_balance=89.60:expected_balance=72.61:actual_balance=44.82:"
        "delta=-27.79"
    ) in failures
    assert (
        "bank_review:missing_row_candidate:"
        "before_row=2:date_range=2022-06-24..2022-07-01:"
        "direction=expense:amount=27.79:balance=61.81:"
        "evidence=balance_chain_only:action=manual_review:not_auto_adopted"
    ) in failures
    assert (
        "bank_review:repair_request:"
        "id=bank-ledger-balance-gap-before-row-2:"
        "kind=missing_ledger_row_local_ocr:"
        "can_render=false:"
        "action=manual_review:"
        "reason=missing_page_bbox"
    ) in failures


@pytest.mark.parametrize(
    ("text", "page_texts"),
    [
        ("账户交易明细\n收支类别：收入", None),
        ("", [(1, "账户交易明细\n收支类别：收入")]),
    ],
    ids=["document-text", "page-business-header"],
)
def test_direction_filtered_export_does_not_report_balance_gap_as_missing_row(
    text: str,
    page_texts: list[tuple[int, str]] | None,
) -> None:
    records = [
        {
            "normalized": {
                "date": "2023-10-02",
                "amount": 23903.69,
                "direction": "income",
                "balance": 23903.69,
            }
        },
        {
            "normalized": {
                "date": "2023-10-07",
                "amount": 13610.09,
                "direction": "income",
                "balance": 13610.09,
            }
        },
    ]

    failures = audit_bank_statement_invariants(records, text, page_texts=page_texts)

    assert not any(item.startswith("bank_invariant_failed:balance_chain") for item in failures)
    assert not any(item.startswith("bank_review:balance_chain_gap") for item in failures)
    assert not any(item.startswith("bank_review:missing_row_candidate") for item in failures)


def test_balance_chain_skips_known_sequence_gap_and_reports_source_page_gap() -> None:
    records = [
        {
            "normalized": {
                "sequence_no": "31",
                "date": "2023-01-07",
                "amount": 20.0,
                "direction": "expense",
                "balance": 80.0,
            }
        },
        {
            "normalized": {
                "sequence_no": "109",
                "date": "2023-01-21",
                "amount": 10.0,
                "direction": "expense",
                "balance": 500.0,
            }
        },
    ]
    text = "第1页共54页 第2页共54页 第7页共54页 第8页共54页"

    warnings = audit_bank_statement_invariants(records, text)

    assert not any(item.startswith("bank_invariant_failed:balance_chain") for item in warnings)
    assert ("bank_review:source_page_gap:observed=4/54:missing_ranges=3-6,9-54:action=manual_review") in warnings


def test_missing_printed_footer_is_suppressed_by_source_provenanced_page_presence() -> None:
    records = [
        {
            "normalized": {
                "date": f"2024-01-{index:02d}",
                "amount": 1.0,
                "direction": "income",
                "balance": float(index),
            },
            "source": {
                "source_page": 1 if index <= 2 else 2,
                "page_range": [1 if index <= 2 else 2, 1 if index <= 2 else 2],
            },
        }
        for index in range(1, 5)
    ]
    page_texts = [(1, "第1页共2页\n序号\n1\n2"), (2, "序号\n3\n4")]

    warnings = audit_bank_statement_invariants(
        records,
        "第1页共2页",
        page_texts=page_texts,
    )

    assert not any(item.startswith("bank_review:source_page_gap") for item in warnings)


def test_missing_printed_footer_still_warns_without_a_source_record_on_every_page() -> None:
    records = [
        {
            "raw": {"序号": "1"},
            "normalized": {"sequence_no": "1", "date": "2024-01-01", "amount": 1.0, "direction": "income"},
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]

    warnings = audit_bank_statement_invariants(
        records,
        "第1页共2页 第1页共2页",
        page_texts=[(1, "第1页共2页"), (2, "page body without a retained row")],
    )

    assert any(item.startswith("bank_review:source_page_gap:observed=1/2") for item in warnings)


def test_contiguous_exported_page_slice_is_not_reported_as_source_gap() -> None:
    records = [
        {
            "normalized": {
                "date": "2024-01-01",
                "amount": 1.0,
                "direction": "income",
            },
            "source": {"source_page": index, "page_range": [index, index]},
        }
        for index in range(1, 4)
    ]
    page_texts = [
        (1, "第4页共9页\n序号/No. 41"),
        (2, "第5页共9页\n序号/No. 42"),
        (3, "第6页共9页\n序号/No. 43"),
    ]

    warnings = audit_bank_statement_invariants(
        records,
        "第4页共9页 第5页共9页 第6页共9页",
        page_texts=page_texts,
    )

    assert not any(item.startswith("bank_review:source_page_gap") for item in warnings)


def test_noncontiguous_exported_page_slice_still_reports_source_gap() -> None:
    records = [
        {
            "raw": {"序号": str(index)},
            "normalized": {"sequence_no": str(index), "date": "2024-01-01", "amount": 1.0, "direction": "income"},
            "source": {"source_page": index, "page_range": [index, index]},
        }
        for index in range(1, 4)
    ]

    warnings = audit_bank_statement_invariants(
        records,
        "第4页共9页 第6页共9页 第7页共9页",
        page_texts=[(1, "第4页共9页"), (2, "第6页共9页"), (3, "第7页共9页")],
    )

    assert any(item.startswith("bank_review:source_page_gap") for item in warnings)


def test_canonical_quality_does_not_mark_partial_rows_success() -> None:
    records = [
        {"normalized": {"date": f"2023-01-{index:02d}", "direction": "expense", "amount": 1.0}}
        for index in range(1, 10)
    ]

    result = audit_cqf(records, canonical_expected=10)

    assert result.coverage_ratio == 0.9
    assert result.extract_status == "low_coverage"


def test_filename_hint_can_route_when_entity_is_a_transaction_channel() -> None:
    parse_result = SimpleNamespace(
        entities=SimpleNamespace(organization="网上银行", domain_specific={}),
        file_path="/tmp/银行流水_中国建设银行_20231228.pdf",
    )

    assert resolve_institution_from_context(parse_result, "网上银行 网银结算") == ("中国建设银行", "filename.token")


def test_counterparty_account_date_prefix_filter_is_calendar_based() -> None:
    for year in (2021, 2024, 2030):
        account, _party = _extract_counterparty(f"交易日期 {year}0101123456 测试对方")
        assert account == ""

    account, _party = _extract_counterparty("对方账号 20241301123456 测试对方")
    assert account == "20241301123456"


def test_bilingual_electronic_statement_identity_after_leading_table() -> None:
    text = "\n".join(
        [
            "|交易日期|发生额|账户余额|",
            *("|2025/01/01|1.00|9.00|" for _ in range(80)),
            "客户名称 Customer Name 重庆某某信用管理有限公司",
            "账户名称 Account Name 重庆某某信用管理有限公司",
            "账号 Account Number 83010078801500000000",
            "账单统计日期 Start Time & End Time 2025/01/01 - 2025/12/31",
            "开户行 The Bank of Account Opening 浦发银行重庆分行营业部",
        ]
    )

    identity = extract_identity_from_header(text)

    assert identity == {
        "account_holder": "重庆某某信用管理有限公司",
        "account_number": "83010078801500000000",
        "query_period": "2025-01-01 ~ 2025-12-31",
        "branch_name": "浦发银行重庆分行营业部",
    }


def test_header_opening_bank_outranks_transaction_body_organization() -> None:
    parse_result = SimpleNamespace(
        entities=SimpleNamespace(organization="重庆农村商业银行", domain_specific={}),
        file_path="/tmp/statement.pdf",
    )
    text = "开户行 The Bank of Account Opening 浦发银行重庆分行营业部\n重庆农村商业银行"

    assert resolve_institution_from_context(parse_result, text) == ("浦发银行重庆分行营业部", "header.branch")


def test_institution_registry_is_owned_by_bank_plugin() -> None:
    assert detect_registered_institution("中国建设银行 账户交易明细 序号 交易日期") == "中国建设银行"
