from __future__ import annotations

import importlib.util

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult, ResultStatus, TextBlock
from docmirror.models.sealed import seal_parse_result
from docmirror.plugin_api import PluginProvider
from docmirror.plugins._runtime.plugin_registry import PluginRegistry


def _sealed(document_type: str = "id_card"):
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        entities=DocumentEntities(
            document_type=document_type,
            domain_specific={"name": "测试", "id_number": "110101199001011234"},
        ),
    )
    return result, seal_parse_result(result)


def test_bundled_community_plugin_runs_only_from_sealed_snapshot() -> None:
    mutable, sealed = _sealed()
    registry = PluginRegistry()
    projector = registry.get_projector("generic", "community", sealed_schema=sealed.schema_version)

    assert projector is not None
    with pytest.raises(TypeError, match="SealedParseResult"):
        projector.project(mutable)

    before = sealed.integrity_fingerprint
    payload = projector.project(sealed)

    assert payload is not None
    assert payload["document"]["type"] == "id_card"
    assert sealed.integrity_fingerprint == before
    assert sealed.verify_integrity()
    assert sealed.to_read_view().entities.domain_specific == {
        "name": "测试",
        "id_number": "110101199001011234",
    }


def test_all_editions_share_one_post_seal_plugin_registry(monkeypatch) -> None:
    class _EnterpriseProjector:
        domain_name = "bank_statement"
        edition = "enterprise"

        def project(self, result):
            return {"edition": self.edition, "fingerprint": result.fact_fingerprint()}

    class _FinanceProjector(_EnterpriseProjector):
        edition = "finance"

    enterprise = _EnterpriseProjector()
    finance = _FinanceProjector()
    monkeypatch.setattr(
        "docmirror.plugins._runtime.discovery.load_plugin_providers",
        lambda: [
            PluginProvider(
                provider_id="test.commercial",
                version="1",
                projectors=(enterprise, finance),
            )
        ],
    )
    registry = PluginRegistry()

    assert registry.list_domains() == {
        "alipay_payment": ["community"],
        "bank_reconciliation": ["community"],
        "bank_statement": ["community", "enterprise", "finance"],
        "business_license": ["community"],
        "credit_report": ["community"],
        "generic": ["community"],
        "vat_invoice": ["community"],
        "wechat_payment": ["community"],
    }
    assert registry.get_projector("bank_statement", "community") is not None
    assert registry.get_projector("bank_statement", "enterprise") is enterprise
    assert registry.get_projector("bank_statement", "finance") is finance


def test_bundled_alias_projector_preserves_document_type_and_reuses_plugin() -> None:
    text = "\n".join(
        [
            "江苏银行对公账户对账单",
            "账户名称：测试有限公司",
            "账号：70650188000156836",
            "借方笔数：1 借方发生总额：2.00 贷方笔数：1 贷方发生总额：5.00 合计笔数：2",
            "序号",
            "交易日期",
            "交易时间",
            "摘要",
            "借方发生额",
            "贷方发生额",
            "余额",
            "对方账户",
            "对方户名",
            "1",
            "2022-01-01",
            "09:00:00",
            "货款",
            "5.00",
            "95.00",
            "2204010309000388825",
            "李四有限公司",
            "2",
            "2022-01-02",
            "10:00:00",
            "收费",
            "2.00",
            "93.00",
            "70650107360000033",
            "企业电子渠道跨行转账手续费收入",
        ]
    )
    sealed = seal_parse_result(
        ParseResult(
            status=ResultStatus.SUCCESS,
            full_text=text,
            pages=[PageContent(page_number=1, texts=[TextBlock(content=text)])],
            entities=DocumentEntities(document_type="bank_reconciliation"),
        )
    )
    projector = PluginRegistry().get_projector(
        "bank_reconciliation",
        "community",
        sealed_schema=sealed.schema_version,
    )

    assert projector is not None
    payload = projector.project(sealed)

    assert payload is not None
    assert payload["document"]["type"] == "bank_reconciliation"
    assert payload["datasets"][0]["row_count"] == 2
    bundle = projector.project_bundle(sealed)
    assert "对账单" in bundle.render_markdown()
    assert sealed.verify_integrity()


@pytest.mark.parametrize(
    "module",
    (
        "docmirror.framework.middlewares.extraction.community_fact_recognizer",
        "docmirror.input.canonical.fact_patch",
        "docmirror.ocr.local_structure.candidate_supplement",
        "docmirror.ocr.micro_grid.materialize",
    ),
)
def test_pre_seal_plugin_bridges_are_retired(module: str) -> None:
    assert importlib.util.find_spec(module) is None
