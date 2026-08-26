"""Post-seal Community projector for financial statements."""

from __future__ import annotations

from docmirror.plugins._base.projector import CommunityProjector, ProjectionData
from docmirror.plugins.financial_statement.projection import derive_financial_statement_projection


class FinancialStatementPlugin(CommunityProjector):
    """Project source-conserving financial statement datasets."""

    @property
    def domain_name(self) -> str:
        return "financial_statement"

    @property
    def display_name(self) -> str:
        return "Financial Statement (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        return derive_financial_statement_projection(parse_result, full_text=text)


plugin = FinancialStatementPlugin()

__all__ = ["FinancialStatementPlugin", "plugin"]
