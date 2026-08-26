"""Post-seal Community projector for tax returns."""

from __future__ import annotations

from docmirror.plugins._base.projector import CommunityProjector, ProjectionData
from docmirror.plugins.tax_return.projection import derive_tax_return_projection


class TaxReturnPlugin(CommunityProjector):
    """Project tax-return identity, sections, and source-order datasets."""

    @property
    def domain_name(self) -> str:
        return "tax_return"

    @property
    def display_name(self) -> str:
        return "Tax Return (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        return derive_tax_return_projection(parse_result, full_text=text)


plugin = TaxReturnPlugin()

__all__ = ["TaxReturnPlugin", "plugin"]
