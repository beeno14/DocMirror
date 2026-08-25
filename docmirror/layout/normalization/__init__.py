"""UDTR page normalization primitives."""

from docmirror.layout.normalization.deskew import (
    HOUGH_DESKEW_METHOD,
    estimate_deskew_angle,
    hough_deskew_image,
    inverse_project_hough_bbox,
)
from docmirror.layout.normalization.models import NormalizationCandidate, NormalizationTrace
from docmirror.layout.normalization.transform import (
    build_identity_trace,
    build_normalization_trace,
    invert_matrix,
    is_invertible_matrix,
    rotation_matrix,
)

__all__ = [
    "NormalizationCandidate",
    "NormalizationTrace",
    "build_identity_trace",
    "build_normalization_trace",
    "estimate_deskew_angle",
    "hough_deskew_image",
    "inverse_project_hough_bbox",
    "HOUGH_DESKEW_METHOD",
    "invert_matrix",
    "is_invertible_matrix",
    "rotation_matrix",
]
