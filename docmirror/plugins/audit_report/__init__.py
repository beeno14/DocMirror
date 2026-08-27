"""Community projection for audit reports."""

from docmirror.plugins.audit_report.community_plugin import AuditReportPlugin, plugin
from docmirror.plugins.audit_report.projection import derive_audit_report_projection

__all__ = ["AuditReportPlugin", "derive_audit_report_projection", "plugin"]
