"""Deskew primitives for the UDTR normalization plane.

The Hough implementation mirrors the proven scanned-credit-report
preprocessor: it only trusts a median supported by at least five long,
near-horizontal or near-vertical rules and applies small corrections in the
closed interval from 0.5 to 5 degrees.  Any weak or malformed input is an
exact no-op.
"""

from __future__ import annotations

from math import atan2, degrees, isfinite
from statistics import median
from typing import Any

HOUGH_DESKEW_METHOD = "hough_lines_p_v1"
HOUGH_DESKEW_RESIZE_MAX_DIM = 1500
HOUGH_DESKEW_THRESHOLD = 80
HOUGH_DESKEW_MIN_LINE_LENGTH = 100
HOUGH_DESKEW_MAX_LINE_GAP = 20
HOUGH_DESKEW_MIN_SUPPORT = 5
HOUGH_DESKEW_MIN_ANGLE = 0.5
HOUGH_DESKEW_MAX_ANGLE = 5.0


def hough_deskew_image(image: Any) -> tuple[Any, dict[str, Any]]:
    """Return a conservatively deskewed image and its affine diagnostics.

    The returned ``forward_matrix`` maps input pixels to output pixels;
    ``inverse_matrix`` performs the inverse mapping.  Like the reference
    preprocessor, the corrected canvas expands with white borders so edge
    content is not clipped.  The forward matrix includes that translation.
    """

    identity = _identity_matrix()
    details: dict[str, Any] = {
        "method": HOUGH_DESKEW_METHOD,
        "applied": False,
        "angle": 0.0,
        "reason": "invalid_image",
        "horizontal_line_count": 0,
        "vertical_line_count": 0,
        "support_line_count": 0,
        "forward_matrix": identity,
        "inverse_matrix": identity,
        "input_width": 0,
        "input_height": 0,
        "output_width": 0,
        "output_height": 0,
    }
    try:
        import cv2
        import numpy as np
    except Exception:
        details["reason"] = "dependency_unavailable"
        return image, details

    try:
        array = np.asarray(image)
        if array.size == 0 or array.ndim not in {2, 3}:
            return image, details
        height, width = array.shape[:2]
        details.update(
            input_width=int(width),
            input_height=int(height),
            output_width=int(width),
            output_height=int(height),
        )
        if height < 50 or width < 50:
            details["reason"] = "image_too_small"
            return image, details

        gray = _hough_gray(array, cv2)
        largest = max(height, width)
        if largest > HOUGH_DESKEW_RESIZE_MAX_DIM:
            scale = HOUGH_DESKEW_RESIZE_MAX_DIM / float(largest)
            gray = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_DESKEW_THRESHOLD,
            minLineLength=HOUGH_DESKEW_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_DESKEW_MAX_LINE_GAP,
        )
        if lines is None or len(lines) == 0:
            details["reason"] = "no_lines"
            return image, details

        horizontal: list[float] = []
        vertical: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = (float(value) for value in line[0])
            dx, dy = x2 - x1, y2 - y1
            if (dx * dx + dy * dy) ** 0.5 < HOUGH_DESKEW_MIN_LINE_LENGTH:
                continue
            angle = degrees(atan2(dy, dx))
            if abs(angle) < 30.0 or abs(angle) > 150.0:
                horizontal.append(angle - (180.0 if angle > 90.0 else -180.0 if angle < -90.0 else 0.0))
            elif abs(abs(angle) - 90.0) < 30.0:
                vertical.append(angle - (90.0 if angle > 0.0 else -90.0))

        deviations = [*horizontal, *vertical]
        details.update(
            horizontal_line_count=len(horizontal),
            vertical_line_count=len(vertical),
            support_line_count=len(deviations),
        )
        if len(deviations) < HOUGH_DESKEW_MIN_SUPPORT:
            details["reason"] = "insufficient_lines"
            return image, details

        angle = float(np.median(deviations))
        if not isfinite(angle):
            details["reason"] = "nonfinite_angle"
            return image, details
        details["detected_angle"] = round(angle, 6)
        if abs(angle) > HOUGH_DESKEW_MAX_ANGLE:
            details["reason"] = "angle_out_of_range"
            return image, details
        if abs(angle) < HOUGH_DESKEW_MIN_ANGLE:
            details["reason"] = "below_threshold"
            return image, details

        from PIL import Image

        pil_image = Image.fromarray(array)
        fill_color = 255 if array.ndim == 2 else tuple([255] * array.shape[2])
        rotated = np.asarray(
            pil_image.rotate(
                angle,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=fill_color,
            )
        ).copy()
        output_height, output_width = rotated.shape[:2]

        # PIL rotates about the input centre and centres that result on the
        # expanded canvas.  Record the equivalent forward affine explicitly
        # so the core can compose it into source provenance and the plugin can
        # inverse-project OCR boxes to its frozen logical-page plane.
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        affine[0][2] += (output_width - width) / 2.0
        affine[1][2] += (output_height - height) / 2.0
        forward = _matrix3_from_affine(affine)
        inverse = _matrix3_from_affine(cv2.invertAffineTransform(affine))
        if not _finite_matrix(forward) or not _finite_matrix(inverse):
            details["reason"] = "transform_invalid"
            return image, details
        details.update(
            applied=True,
            angle=round(angle, 6),
            reason="applied",
            forward_matrix=forward,
            inverse_matrix=inverse,
            output_width=output_width,
            output_height=output_height,
        )
        return rotated, details
    except Exception as exc:
        details.update(reason="error", error_type=type(exc).__name__)
        return image, details


def inverse_project_hough_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    details: dict[str, Any],
    *,
    width: float,
    height: float,
) -> list[float] | None:
    """Map one deskewed OCR AABB back to the input logical-image plane."""

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        return None
    matrix = details.get("inverse_matrix") if details.get("applied") is True else _identity_matrix()
    if not _finite_matrix(matrix):
        return None
    points = [
        _apply_matrix(matrix, values[0], values[1]),
        _apply_matrix(matrix, values[2], values[1]),
        _apply_matrix(matrix, values[2], values[3]),
        _apply_matrix(matrix, values[0], values[3]),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    out = [max(0.0, min(xs)), max(0.0, min(ys)), min(float(width), max(xs)), min(float(height), max(ys))]
    return out if out[2] > out[0] and out[3] > out[1] else None


def _hough_gray(array: Any, cv2: Any) -> Any:
    if array.ndim == 2:
        return array
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def _identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _matrix3_from_affine(matrix: Any) -> list[list[float]]:
    return [
        [float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])],
        [float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])],
        [0.0, 0.0, 1.0],
    ]


def _finite_matrix(matrix: Any) -> bool:
    return bool(
        isinstance(matrix, list)
        and len(matrix) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in matrix)
        and all(isfinite(float(value)) for row in matrix for value in row)
    )


def _apply_matrix(matrix: list[list[float]], x: float, y: float) -> tuple[float, float]:
    return (
        float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2]),
        float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2]),
    )


def estimate_deskew_angle(page_or_image: Any) -> float:
    """Return a conservative page deskew angle in degrees.

    The estimator intentionally prefers high-precision, low-recall signals:
    vector line segments first, OCR/text baselines second, and image Hough lines
    last when OpenCV/numpy are available. It returns ``0.0`` whenever evidence is
    weak, because an incorrect deskew is more damaging than a missed one.
    """

    angles = [
        *_angles_from_vector_lines(page_or_image),
        *_angles_from_text_bboxes(page_or_image),
        *_angles_from_image(page_or_image),
    ]
    near_horizontal = [angle for angle in angles if -15.0 <= angle <= 15.0 and abs(angle) >= 0.05]
    if len(near_horizontal) < 2:
        return 0.0
    estimate = float(median(near_horizontal))
    if not isfinite(estimate) or abs(estimate) > 5.0:
        return 0.0
    return round(estimate, 4)


def _angles_from_vector_lines(page_or_image: Any) -> list[float]:
    lines = _iter_items(page_or_image, ("vector_lines", "lines", "edges"))
    out: list[float] = []
    for line in lines:
        x0, y0, x1, y1 = _line_points(line)
        if x0 is None:
            continue
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        if abs(dx) < 24 or abs(dx) < abs(dy) * 3:
            continue
        out.append(_normalize_angle(degrees(atan2(dy, dx))))
    return out


def _angles_from_text_bboxes(page_or_image: Any) -> list[float]:
    tokens = _iter_items(page_or_image, ("tokens", "ocr_tokens", "text_atoms", "texts"))
    boxes: list[list[float]] = []
    for token in tokens:
        bbox = _bbox(token)
        if bbox and len(bbox) >= 4:
            boxes.append([float(v) for v in bbox[:4]])
    if len(boxes) < 4:
        return []
    boxes.sort(key=lambda box: ((box[1] + box[3]) / 2.0, box[0]))
    heights = [max(1.0, box[3] - box[1]) for box in boxes]
    line_tolerance = max(3.0, median(heights) * 0.8)
    lines: list[list[list[float]]] = []
    for box in boxes:
        cy = (box[1] + box[3]) / 2.0
        if not lines:
            lines.append([box])
            continue
        previous_cy = median([(item[1] + item[3]) / 2.0 for item in lines[-1]])
        if abs(cy - previous_cy) <= line_tolerance:
            lines[-1].append(box)
        else:
            lines.append([box])

    out: list[float] = []
    for line in lines:
        if len(line) < 3:
            continue
        first, last = min(line, key=lambda box: box[0]), max(line, key=lambda box: box[2])
        dx = ((last[0] + last[2]) / 2.0) - ((first[0] + first[2]) / 2.0)
        dy = ((last[1] + last[3]) / 2.0) - ((first[1] + first[3]) / 2.0)
        if abs(dx) >= 40:
            out.append(_normalize_angle(degrees(atan2(dy, dx))))
    return out


def _angles_from_image(page_or_image: Any) -> list[float]:
    image = _image_array(page_or_image)
    if image is None:
        return []
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            return []
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_len = max(32, int(min(gray.shape[:2]) * 0.15))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=12)
        if lines is None:
            return []
        out: list[float] = []
        for item in lines[:200]:
            x0, y0, x1, y1 = [float(v) for v in item[0]]
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) >= min_len and abs(dx) >= abs(dy) * 3:
                out.append(_normalize_angle(degrees(atan2(dy, dx))))
        return out
    except Exception:
        return []


def _iter_items(source: Any, names: tuple[str, ...]) -> list[Any]:
    for name in names:
        value = _get(source, name)
        if isinstance(value, list | tuple):
            return list(value)
    if isinstance(source, list | tuple):
        return list(source)
    return []


def _line_points(line: Any) -> tuple[float | None, float | None, float | None, float | None]:
    if isinstance(line, dict):
        if {"x0", "y0", "x1", "y1"} <= set(line):
            return float(line["x0"]), float(line["y0"]), float(line["x1"]), float(line["y1"])
        if isinstance(line.get("bbox"), list | tuple) and len(line["bbox"]) >= 4:
            x0, y0, x1, y1 = line["bbox"][:4]
            return float(x0), float(y0), float(x1), float(y1)
    if isinstance(line, list | tuple) and len(line) >= 4:
        x0, y0, x1, y1 = line[:4]
        return float(x0), float(y0), float(x1), float(y1)
    return None, None, None, None


def _bbox(item: Any) -> list[float] | None:
    value = _get(item, "bbox")
    if isinstance(value, list | tuple) and len(value) >= 4:
        return [float(v) for v in value[:4]]
    if isinstance(item, list | tuple) and len(item) >= 4:
        return [float(v) for v in item[:4]]
    return None


def _image_array(source: Any) -> Any:
    for name in ("image", "array", "pixels"):
        value = _get(source, name)
        if value is not None:
            return value
    return source if _looks_like_image(source) else None


def _get(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _looks_like_image(source: Any) -> bool:
    shape = getattr(source, "shape", None)
    return isinstance(shape, tuple) and len(shape) >= 2


def _normalize_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle
