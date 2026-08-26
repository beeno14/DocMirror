# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic community plugin tests."""

from __future__ import annotations

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult, ResultStatus
from docmirror.plugins.generic.community_plugin import plugin


def _mirror(document_type: str, domain_specific: dict | None = None) -> ParseResult:
    pr = ParseResult(status=ResultStatus.SUCCESS)
    pr.entities = DocumentEntities(
        document_type=document_type,
        domain_specific=domain_specific or {},
    )
    return pr


def test_generic_plugin_domain_name():
    assert plugin.domain_name == "generic"
    assert callable(plugin.project)
    assert callable(plugin.derive)


def test_id_card_classified_produces_generic_facts():
    patch = plugin.derive(
        _mirror("id_card", {"name": "张三", "id_number": "110101199001011234"})
    )
    assert patch.projector_id == "generic"
    assert patch.document_type == "id_card"
    assert patch.domain_facts["name"] == "张三"


def test_generic_output_collects_key_values():
    pr = _mirror("payroll_slip")
    kv = type("KV", (), {"key": "姓名", "value": "李四", "confidence": 1.0, "bbox": None, "evidence_ids": None})()
    page = type("Page", (), {"key_values": [kv], "tables": [], "texts": [], "page_number": 1, "width": 800, "height": 1000})()
    pr.pages = [page]

    patch = plugin.derive(pr)
    assert patch.domain_facts["姓名"] == "李四"


def test_generic_output_rejects_internal_and_serialized_table_key_values():
    pr = _mirror("tax_return")
    kv_type = type("KV", (), {"confidence": 1.0, "bbox": None, "evidence_ids": None})
    valid = kv_type()
    valid.key, valid.value = "纳税人识别号", "91330100TEST"
    internal = kv_type()
    internal.key, internal.value = "mirror_ltqg_passed_tables", "7"
    serialized = kv_type()
    serialized.key, serialized.value = "纳税人名称", "测试公司\t法定代表人：张三"
    page = type(
        "Page",
        (),
        {"key_values": [valid, internal, serialized], "tables": [], "texts": [], "page_number": 1},
    )()
    pr.pages = [page]

    patch = plugin.derive(pr)

    assert patch.domain_facts["纳税人识别号"] == "91330100TEST"
    assert "mirror_ltqg_passed_tables" not in patch.domain_facts
    assert "纳税人名称" not in patch.domain_facts


def test_generic_output_replaces_invalid_name_fragment_with_complete_page_kv():
    pr = _mirror("tax_return", {"extracted_entities": {"纳税人名称": "(公"}})
    kv = type(
        "KV",
        (),
        {
            "key": "纳税人名称",
            "value": "上海测试有限公司",
            "confidence": 0.95,
            "bbox": None,
            "evidence_ids": None,
        },
    )()
    page = type(
        "Page",
        (),
        {"key_values": [kv], "tables": [], "texts": [], "page_number": 1, "width": 800, "height": 1000},
    )()
    pr.pages = [page]

    patch = plugin.derive(pr)

    assert patch.domain_facts["纳税人名称"] == "上海测试有限公司"


def test_generic_projection_does_not_mutate_parse_result():
    pr = _mirror("expense_report", {"报销单号": "BX-001", "金额": "1,000.00"})
    before = pr.model_dump(mode="python")

    plugin.derive(pr, "部门：销售部")

    assert pr.model_dump(mode="python") == before
