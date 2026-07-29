# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvidenceEngine payment ledger disambiguation."""

from __future__ import annotations

from docmirror.layout.scene.evidence_engine import EvidenceEngine
from docmirror.models.entities.parse_result import (
    CanonicalEvidencePlane,
    CellValue,
    DocumentEntities,
    PageContent,
    ParseResult,
    TableBlock,
    TableRow,
    TextBlock,
    TextLevel,
)
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore


def _alipay_header_table() -> TableBlock:
    headers = [
        "收/支",
        "交易对方",
        "商品说明",
        "收/付款方式",
        "金额",
        "交易订单号",
        "商家订单号",
        "交易时间",
    ]
    row = TableRow(cells=[CellValue(text="支出") for _ in headers])
    return TableBlock(table_id="t1", headers=headers, rows=[row])


def _wechat_header_table() -> TableBlock:
    headers = ["交易单号", "交易时间", "交易类型", "收/支", "金额"]
    row = TableRow(cells=[CellValue(text="支出") for _ in headers])
    return TableBlock(table_id="t1", headers=headers, rows=[row])


def _bank_monthly_statement_table() -> TableBlock:
    headers = ["交易日期", "柜员流水号", "发生额", "账户余额", "交易对手信息", "摘要代码", "备注"]
    row = TableRow(cells=[CellValue(text="test") for _ in headers])
    return TableBlock(table_id="t1", headers=headers, rows=[row])


def _titleless_bank_table(row_count: int = 10) -> TableBlock:
    rows = []
    for index in range(row_count):
        rows.append(
            TableRow(
                cells=[
                    CellValue(text=f"202303{index + 1:02d} 10:20:30"),
                    CellValue(text=f"53701202303221584326182544{index:02d}"),
                    CellValue(text=f"{1000 + index:,}.00"),
                    CellValue(text=f"{5000 + index:,}.00"),
                    CellValue(text=f"62262238030761{index:02d}"),
                    CellValue(text="中国民生银行股份有限公司"),
                ]
            )
        )
    return TableBlock(table_id="continuation", rows=rows)


def test_bank_document_frame_survives_payment_channel_keyword_veto():
    document_text = "\n".join(
        [
            "常熟农村商业银行",
            "账号：6214****1234",
            "起止日期：2023-01-01 至 2023-06-30",
            "交易明细",
            "余额",
            "对方户名",
            "收入/支出金额",
            "微信支付 财付通支付科技有限公司",
            "支付宝（中国）网络技术有限公司",
        ]
    )
    result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=document_text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_statement"
    evidence_log = classified.entities.domain_specific["classification_provenance"]
    assert evidence_log["document_type"] == "bank_statement"


def test_bank_issuer_account_and_headers_outrank_payment_body_keywords():
    document_text = "\n".join(
        [
            "建设银行- 张白华",
            "卡号/账号:6217001300005799744",
            "客户名称：张白华",
            "序号 摘要 币别 钞汇 交易日期 交易金额 账户余额 交易地点/附言 对方账号与户名",
            "消费 支付宝-天弘基金管理有限公司 财付通-微信支付-拼多多平台商户",
            "支付宝（中国）网络技术有限公司 扫二维码付款 微信支付",
            "20220611 -5.88 92.28 105331000000875/支付宝-天弘基金管理有限公司",
        ]
    )
    result = ParseResult(
        full_text=document_text,
        pages=[PageContent(page_number=1, texts=[TextBlock(content=document_text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_statement"


def test_weak_bank_terms_do_not_protect_bank_statement():
    engine = EvidenceEngine()
    evidence = engine._text_frame_evidence(
        "微信支付 银行 账号 交易明细 余额",
        "",
    )

    assert "bank_statement" not in engine._protected_document_categories(evidence)


def test_bank_identity_outranks_payment_voucher_body_words():
    document_text = "\n".join(
        [
            "中国建设银行股份有限公司活期存款明细账",
            "日期 20230101-20230131",
            "账号 31050184390000001412",
            "账户名称 上海丰营供应链管理有限公司",
            "币别 人民币元",
            "打印时间 20230208",
            "日期 发生额 借贷 余额 摘要 对方户名",
            "电子回单 转账凭证 付款",
        ]
    )
    result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=document_text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    assert EvidenceEngine().process(result).entities.document_type == "bank_statement"


def test_bank_identity_outranks_wechat_transaction_body():
    document_text = "\n".join(
        [
            "招商银行交易流水",
            "Name 陈志侃",
            "Account No 6214832149628732",
            "开户行 上海分行丽园支行",
            "申请时间 2022-06-23",
            "记账日期 Currency 交易金额 联机余额 交易摘要 对手信息",
            "财付通-微信支付 快捷支付 微信支付",
        ]
    )
    result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=document_text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    assert EvidenceEngine().process(result).entities.document_type == "bank_statement"


def test_bank_monthly_statement_survives_embedded_swift_code_prefix():
    document_text = "\n".join(
        [
            "上海浦东发展银行",
            "客户名称 上海煦宝实业有限公司",
            "账户名称 上海煦宝实业有限公司",
            "账号 98410154740005738",
            "账单币种 人民币",
            "账单类型 月账单",
            "交易日期 柜员流水号 发生额 账户余额 交易对手信息 摘要代码 备注",
            "2023年08月31日25D06ALPHHEL7MT2023年9月1日",
        ]
    )
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content=document_text)],
                tables=[_bank_monthly_statement_table()],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_statement"


def test_embedded_mt202_prefix_is_not_swift_evidence():
    evidence = EvidenceEngine()._keyword_evidence("25D06ALPHHEL7MT2023年9月1日")

    assert all(item.category != "swift_message" for item in evidence)


def test_standalone_mt202_remains_swift_evidence():
    evidence = EvidenceEngine()._keyword_evidence("SWIFT payment message: MT202")

    assert any(item.category == "swift_message" for item in evidence)


def test_wechat_statement_with_bank_transfer_text_stays_wechat():
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(content="微信支付交易明细证明", level=TextLevel.TITLE),
                    TextBlock(content="财付通支付科技有限公司\n银行卡 银行转账"),
                ],
                tables=[_wechat_header_table()],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "wechat_payment"


def test_alipay_headers_prefer_alipay_over_wechat():
    engine = EvidenceEngine()
    result = ParseResult(
        pages=[PageContent(page_number=1, tables=[_alipay_header_table()])],
    )
    result.entities.domain_specific = {
        "extractor_scene_hint": "alipay_payment",
        "extractor_scene_confidence": 0.99,
    }
    classified = engine.process(result)
    assert classified.entities.document_type == "alipay_payment"


def test_alipay_statement_without_hint_stays_alipay():
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(content="支付宝（中国）网络技术有限公司", level=TextLevel.TITLE),
                    TextBlock(content="支付宝交易明细\n银行卡付款"),
                ],
                tables=[_alipay_header_table()],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "alipay_payment"


def test_nfkc_normalizes_compatibility_glyphs_in_bank_title():
    text = "\n".join(
        [
            "兴业银⾏交易流⽔",
            "户名 郑萍杰",
            "账号 622908166157848113",
            "币种 人民币",
            "交易⽇期 交易⾦额 账户余额 摘要 对方户名",
            "打印日期 2024-02-20",
        ]
    )
    result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    assert EvidenceEngine().process(result).entities.document_type == "bank_statement"


def test_personal_account_reconciliation_is_emitted_as_bank_statement():
    text = "\n".join(
        [
            "个人账户对账单",
            "客户姓名 孙嘉蔚",
            "客户账号 6226223803076168",
            "开户机构 中国民生银行股份有限公司",
            "起止日期 2023/01/22 - 2024/01/22",
            "币种 人民币",
            "交易时间 摘要 交易金额 账户余额 对方户名",
        ]
    )
    result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_statement"
    assert classified.entities.document_type != "bank_reconciliation"


def test_corporate_account_reconciliation_is_emitted_as_bank_reconciliation():
    text = "\n".join(
        [
            "江苏银行对公账户对账单",
            "账户名称 镇江东翔网络科技有限公司",
            "账号 70650188000202939",
            "起始日期 2023-06-01 终止日期 2023-06-13",
            "借方笔数 11 借方发生总额 23,859.51 贷方笔数 14 贷方发生总额 23,077.85 合计笔数 25",
            "序号 交易日期 交易时间 摘要 借方发生额 贷方发生额 余额 对方账户 对方户名",
        ]
    )
    result = ParseResult(
        full_text=text,
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_reconciliation"
    assert classified.entities.domain_specific["canonical_document_type"] == "bank_statement"


def test_corporate_electronic_statement_controls_are_bank_reconciliation_without_native_title_text():
    text = "\n".join(
        [
            "版面占位字符" * 900,
            "客户名称 Customer Name 重庆某某信用管理有限公司",
            "账户名称 Account Name 重庆某某信用管理有限公司",
            "账号 Account Number 83010078801500000000",
            "账单统计日期 Start Time & End Time 2025/01/01-2025/12/31",
            "汇总交易笔数 Total number of transactions 174笔",
            "借方发生总额 The Total Debit Amount 1,047,323.06",
            "贷方发生总额 The Total Credit Amount 1,063,069.78",
            "交易日期 Transaction Date 发生额 Transaction Amount 账户余额 Account Balance",
            "借方 Debit 贷方 Credit 交易对手信息 Counterparty Information",
        ]
    )
    result = ParseResult(
        full_text=text,
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_reconciliation"
    assert classified.entities.domain_specific["canonical_document_type"] == "bank_statement"


def test_visible_electronic_statement_title_from_evidence_atoms_sets_reconciliation_alias():
    text = "\n".join(
        [
            "客户名称 Customer Name 重庆某某信用管理有限公司",
            "账号 Account Number 83010078801500000000",
            "交易日期 Transaction Date 发生额 Transaction Amount 账户余额 Account Balance",
        ]
    )
    result = ParseResult(
        full_text=text,
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
        entities=DocumentEntities(document_type="unknown"),
        evidence_plane=CanonicalEvidencePlane(
            evidence=EvidenceStore(
                text_atoms=[
                    EvidenceAtom(
                        id="ev:0001:text:000001",
                        kind="text_token",
                        source_kind="pdf_native",
                        page_id="page:0001",
                        text="上海浦东发展银行电子对账单",
                        bbox=[10.0, 10.0, 300.0, 30.0],
                    ),
                    EvidenceAtom(
                        id="ev:0001:text:000002",
                        kind="text_token",
                        source_kind="pdf_native",
                        page_id="page:0001",
                        text="汇总交易笔数 174 借方发生总额 1,047,323.06 贷方发生总额 1,063,069.78",
                        bbox=[10.0, 40.0, 500.0, 60.0],
                    ),
                ]
            )
        ),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "bank_reconciliation"
    assert classified.entities.domain_specific["canonical_document_type"] == "bank_statement"


def test_title_region_prefers_page_geometry_over_reading_order():
    noise = [
        TextBlock(
            content=f"正文噪声 {index}",
            reading_order=index,
            bbox=[20, 500 + index * 10, 300, 508 + index * 10],
        )
        for index in range(15)
    ]
    title = TextBlock(
        content=(
            "银行交易流水 账号 622908166157848113 户名 郑萍杰 币种 人民币 "
            "交易日期 交易金额 账户余额 摘要 对方户名"
        ),
        reading_order=99,
        bbox=[40, 30, 500, 90],
    )
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                height=800,
                texts=[*noise, title],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    assert EvidenceEngine().process(result).entities.document_type == "bank_statement"


def test_titleless_continuation_ledger_uses_strict_structure_fallback():
    engine = EvidenceEngine()
    result = ParseResult(
        pages=[
            PageContent(page_number=1, tables=[_titleless_bank_table(5)]),
            PageContent(page_number=2, tables=[_titleless_bank_table(5)]),
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    structure_evidence = engine._bank_ledger_structure_evidence(result, "")
    classified = engine.process(result)

    assert structure_evidence[0].source == "bank_ledger_structure"
    assert classified.entities.document_type == "bank_statement"


def test_payment_platform_title_blocks_bank_structure_fallback():
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="微信支付交易明细证明", level=TextLevel.TITLE)],
                tables=[_titleless_bank_table(10), _wechat_header_table()],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    classified = EvidenceEngine().process(result)

    assert classified.entities.document_type == "wechat_payment"


def test_sparse_transaction_rows_do_not_confirm_bank_statement():
    result = ParseResult(
        pages=[PageContent(page_number=1, tables=[_titleless_bank_table(2)])],
        entities=DocumentEntities(document_type="unknown"),
    )

    evidence = EvidenceEngine()._bank_ledger_structure_evidence(result, "")

    assert evidence == []


def test_ocr_spaced_date_and_amounts_count_as_continuation_row():
    row = [
        "202303 22 19:37:22",
        "53701202303221584326182544725C11",
        "",
        "731, 712. 00",
        "1, 903, 489. 93",
        "九江银行 股份有限 公司",
    ]

    assert EvidenceEngine._is_bank_continuation_row(row) is True


def test_garbled_scan_without_identity_or_structure_stays_generic():
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                page_mode="scanned",
                texts=[TextBlock(content="真田华省区 C0 00008099 华区补工滨货 送R 00000°09")],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )

    assert EvidenceEngine().process(result).entities.document_type == "generic"
