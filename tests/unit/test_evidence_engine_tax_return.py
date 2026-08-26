# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tax-return identity must outrank shared business-document fields."""

from __future__ import annotations

import pytest

from docmirror.layout.scene.evidence_engine import EvidenceEngine
from docmirror.models.entities.parse_result import (
    DocumentEntities,
    PageContent,
    ParseResult,
    TextBlock,
    TextLevel,
)


def _classify(text: str) -> str:
    result = ParseResult(
        full_text=text,
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content=text, level=TextLevel.TITLE)],
            )
        ],
        entities=DocumentEntities(document_type="unknown"),
    )
    return EvidenceEngine().process(result).entities.document_type


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(
            [
                "增值税及附加税费申报表（一般纳税人适用）",
                "纳税人识别号 91320000TEST000001",
                "税款所属期 2025年1月1日至2025年1月31日",
                "统一社会信用代码 91320000TEST000001",
                "开户银行及账号 测试银行 000000000000000000",
                "增值税专用发票销售额 销项税额 应纳税额 本期应补退税额",
            ]
        ),
        "\n".join(
            [
                "增值税纳税申报表",
                "小规模纳税人适用 税款所属期 2024年第四季度",
                "纳税人识别号 91320000TEST000002",
                "资产负债表 流动资产 资产总计 负债合计",
                "利润表 营业收入 营业成本 利润总额",
                "现金流量表 经营活动产生的现金流量",
            ]
        ),
    ],
)
def test_tax_return_title_outranks_shared_identity_and_attachment_terms(text: str) -> None:
    assert _classify(text) == "tax_return"


@pytest.mark.parametrize(
    ("expected", "text"),
    [
        (
            "business_license",
            "\n".join(
                [
                    "营业执照",
                    "统一社会信用代码 91320000TEST000003",
                    "名称 测试科技有限公司",
                    "类型 有限责任公司",
                    "法定代表人 测试人员",
                    "注册资本 成立日期 营业期限 经营范围 登记机关",
                ]
            ),
        ),
        (
            "tax_certificate",
            "\n".join(
                [
                    "税收完税证明",
                    "纳税人识别号 91320000TEST000004",
                    "税务机关 测试税务局",
                    "实缴金额 100.00",
                ]
            ),
        ),
        (
            "bank_statement",
            "\n".join(
                [
                    "测试银行账户交易明细",
                    "账户名称 测试科技有限公司",
                    "账号 000000000000000005",
                    "起止日期 2025-01-01 至 2025-01-31",
                    "交易日期 交易金额 账户余额 摘要 对方户名",
                ]
            ),
        ),
        (
            "vat_invoice",
            "\n".join(
                [
                    "增值税专用发票",
                    "发票代码 000000000000 发票号码 00000000 开票日期 2025年1月1日",
                    "购方名称 测试购买方 销方名称 测试销售方",
                    "货物或应税劳务名称 金额 税率 税额 价税合计",
                ]
            ),
        ),
    ],
)
def test_tax_return_rules_preserve_other_document_identities(expected: str, text: str) -> None:
    assert _classify(text) == expected
