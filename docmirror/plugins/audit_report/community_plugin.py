"""Post-seal Community projector for audit reports."""

from __future__ import annotations

from typing import Any

from docmirror.output.community_bundle import CommunityBundle
from docmirror.plugins._base.projector import CommunityProjector, ProjectionData
from docmirror.plugins.audit_report.projection import derive_audit_report_projection
from docmirror.plugins.audit_report.reading_projection import render_audit_reading_markdown


class _AuditReportCommunityBundle(CommunityBundle):
    """Keep audit reading cleanup inside the audit plugin boundary."""

    def render_enhanced_markdown(self, semantic: dict[str, Any] | None = None) -> str:
        return render_audit_reading_markdown(semantic or self.semantic_payload())


class AuditReportPlugin(CommunityProjector):
    """Project audit metadata, report structure, and source-backed tables."""

    @property
    def domain_name(self) -> str:
        return "audit_report"

    @property
    def display_name(self) -> str:
        return "Audit Report (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        return derive_audit_report_projection(parse_result, full_text=text)

    def project_bundle(
        self,
        sealed: Any,
        *,
        file_path: str = "",
        file_id: str = "001",
        document_id: str = "",
    ) -> CommunityBundle | None:
        """Build an audit-policy bundle with an audit-only enhanced renderer."""

        from docmirror.models.sealed import SealedParseResult
        from docmirror.output.community_bundle import project_community_bundle
        from docmirror.plugins._base.projector import load_projection_policy

        if not isinstance(sealed, SealedParseResult):
            raise TypeError(f"{type(self).__name__}.project expects SealedParseResult")
        if not self.supports(sealed):
            return None
        before = sealed.integrity_fingerprint
        read_view = sealed.to_read_view()
        derived = self.derive(read_view, str(read_view.full_text or read_view.raw_text or ""))
        projected = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=load_projection_policy(type(self).__module__.rsplit(".", 1)[0]),
        )
        units = {
            key: str(derived.domain_facts[key])
            for key in ("currency", "currency_unit")
            if derived.domain_facts.get(key) not in (None, "")
        }
        if units:
            projected.document["units"] = units
        bundle = _AuditReportCommunityBundle(
            schema=projected.schema,
            document=projected.document,
            sections=projected.sections,
            datasets=projected.datasets,
            files=projected.files,
            warnings=projected.warnings,
            result=projected.result,
            source_fingerprint=projected.source_fingerprint,
            parse_result_schema=projected.parse_result_schema,
            classification=projected.classification,
            domain=projected.domain,
            diagnostics=projected.diagnostics,
            content_markdown_override=projected.content_markdown_override,
        )
        bundle.render_markdown()
        if sealed.integrity_fingerprint != before or not sealed.verify_integrity():
            raise RuntimeError("Post-seal projector changed the sealed snapshot")
        return bundle


plugin = AuditReportPlugin()

__all__ = ["AuditReportPlugin", "plugin"]
