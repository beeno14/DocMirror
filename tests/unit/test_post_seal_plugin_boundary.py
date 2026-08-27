from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult, ResultStatus, TextBlock
from docmirror.models.sealed import seal_parse_result
from docmirror.plugin_api import PluginProvider
from docmirror.plugins._runtime import plugin_registry as plugin_registry_module
from docmirror.plugins._runtime.plugin_registry import PluginRegistry
from docmirror.server.output_builder import build_community_bundle


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


def test_missing_domain_projector_uses_parse_result_fallback() -> None:
    _mutable, sealed = _sealed("future_domain")

    bundle = build_community_bundle(sealed)
    semantic = bundle.semantic_payload()

    assert bundle.schema["support_level"] == "generic"
    assert semantic["classification"] == {
        "document_type": "future_domain",
        "projector_id": "parse_result_fallback",
        "support_level": "generic",
        "confidence": 1.0,
        "reason": "post-seal generic projection",
        "fallback_reason": "community_projector_not_registered",
        "fallback_from_document_type": "future_domain",
    }
    assert semantic["diagnostics"]["community_fallback"] == {
        "reason": "community_projector_not_registered",
        "document_type": "future_domain",
        "source": "sealed_parse_result",
    }
    assert any(
        warning["code"] == "COMMUNITY_PARSE_RESULT_FALLBACK"
        and "community_projector_not_registered" in warning["message"]
        for warning in bundle.warnings
    )


def test_projector_returning_none_uses_parse_result_fallback(monkeypatch) -> None:
    class _UnavailableProjector:
        domain_name = "future_domain"
        edition = "community"

        @staticmethod
        def project_bundle(*_args, **_kwargs):
            return None

    from docmirror.plugins._runtime.plugin_registry import registry

    unavailable = _UnavailableProjector()
    original_get_projector = registry.get_projector

    def get_projector(domain_name, edition, *, sealed_schema=None):
        if domain_name == "future_domain" and edition == "community":
            return unavailable
        return original_get_projector(domain_name, edition, sealed_schema=sealed_schema)

    monkeypatch.setattr(registry, "get_projector", get_projector)
    _mutable, sealed = _sealed("future_domain")

    bundle = build_community_bundle(sealed)

    assert bundle.classification["projector_id"] == "parse_result_fallback"
    assert bundle.classification["fallback_reason"] == "community_projector_returned_none"


def test_bundled_provider_import_failure_does_not_disable_generic_fallback(monkeypatch) -> None:
    original_import_module = plugin_registry_module.importlib.import_module

    def import_module(name: str, package=None):
        if name == "docmirror.plugins.credit_report.community_plugin":
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr(plugin_registry_module.importlib, "import_module", import_module)
    registry = PluginRegistry()

    assert registry.get_projector("generic", "community") is not None
    assert registry.get_projector("credit_report", "community") is None
    assert registry.get_projector("bank_statement", "community") is not None


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
        "audit_report": ["community"],
        "balance_sheet": ["community"],
        "bank_reconciliation": ["community"],
        "bank_statement": ["community", "enterprise", "finance"],
        "business_license": ["community"],
        "cash_flow_statement": ["community"],
        "credit_report": ["community"],
        "financial_report": ["community"],
        "financial_statement": ["community"],
        "generic": ["community"],
        "income_statement": ["community"],
        "owners_equity_changes": ["community"],
        "tax_return": ["community"],
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


def test_core_scanned_extraction_has_no_financial_or_tax_semantics() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "docmirror/input/extraction/extractor.py",
        root / "docmirror/input/extraction/scanned_table_reconstructor.py",
    ]
    forbidden = (
        "docmirror.plugins",
        "financial_statement",
        "tax_return",
        "资产负债表",
        "利润表",
        "现金流量表",
        "纳税申报",
        "一般项目",
        "即征即退",
    )

    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden), path

    generic_adapter = (root / "docmirror/plugins/_base/generic_community_adapter.py").read_text(encoding="utf-8")
    assert "_FINANCIAL_PERIOD_HEADER_RE" not in generic_adapter
    assert "_trim_trailing_statement_noise" not in generic_adapter
