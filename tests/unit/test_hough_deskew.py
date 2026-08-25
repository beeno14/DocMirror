from types import SimpleNamespace

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from docmirror.input.extraction import extractor as extractor_module
from docmirror.layout.normalization import hough_deskew_image, inverse_project_hough_bbox
from docmirror.models.mirror.vnext import EvidenceAtom
from docmirror.quality.udtr_gates import build_udtr_quality_gates


def _ruled_page(*, angle: float = 0.0, lines: int = 10):
    image = np.full((720, 520, 3), 255, dtype=np.uint8)
    for index in range(lines):
        y = 70 + index * 52
        cv2.line(image, (45, y), (475, y), (20, 20, 20), 3)
    for x in (45, 180, 320, 475):
        cv2.line(image, (x, 45), (x, 650), (20, 20, 20), 2)
    if abs(angle) < 1e-9:
        return image
    matrix = cv2.getRotationMatrix2D((260.0, 360.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (520, 720),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


@pytest.mark.parametrize("applied", [-1.5, 1.5])
def test_hough_deskew_corrects_small_table_skew_and_expands_white_canvas(applied):
    image = _ruled_page(angle=applied)

    corrected, audit = hough_deskew_image(image)
    _second, residual = hough_deskew_image(corrected)

    assert audit["applied"] is True
    assert audit["angle"] == pytest.approx(-applied, abs=0.25)
    assert audit["support_line_count"] >= 5
    assert corrected.shape[0] > image.shape[0]
    assert corrected.shape[1] > image.shape[1]
    assert corrected.dtype == image.dtype
    assert audit["input_width"] == image.shape[1]
    assert audit["input_height"] == image.shape[0]
    assert audit["output_width"] == corrected.shape[1]
    assert audit["output_height"] == corrected.shape[0]
    assert int(corrected[0, 0].min()) >= 250
    assert residual["applied"] is False
    assert abs(float(residual.get("detected_angle") or 0.0)) < 0.5


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        (np.full((500, 400, 3), 255, dtype=np.uint8), "no_lines"),
        (_ruled_page(angle=0.2), "below_threshold"),
        (_ruled_page(angle=6.0), "angle_out_of_range"),
        (cv2.line(np.full((500, 400, 3), 255, dtype=np.uint8), (40, 250), (360, 250), (20, 20, 20), 2), "insufficient_lines"),
    ],
)
def test_hough_deskew_fail_closed_cases_are_exact_noops(image, reason):
    corrected, audit = hough_deskew_image(image)

    assert corrected is image
    assert audit["applied"] is False
    assert audit["angle"] == 0.0
    assert audit["reason"] == reason


def test_inverse_project_hough_bbox_restores_original_logical_plane():
    image = _ruled_page(angle=1.5)
    _corrected, audit = hough_deskew_image(image)
    original_bbox = [100.0, 140.0, 260.0, 180.0]
    matrix = audit["forward_matrix"]
    points = [
        (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        )
        for x, y in (
            (original_bbox[0], original_bbox[1]),
            (original_bbox[2], original_bbox[1]),
            (original_bbox[2], original_bbox[3]),
            (original_bbox[0], original_bbox[3]),
        )
    ]
    deskewed_bbox = [
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    ]

    restored = inverse_project_hough_bbox(deskewed_bbox, audit, width=520, height=720)

    assert restored is not None
    assert restored[0] <= original_bbox[0]
    assert restored[1] <= original_bbox[1]
    assert restored[2] >= original_bbox[2]
    assert restored[3] >= original_bbox[3]


def test_hough_deskew_uses_reference_pillow_expand_bicubic_rotation():
    image = _ruled_page(angle=1.5)

    corrected, audit = hough_deskew_image(image)
    expected = np.asarray(
        Image.fromarray(image).rotate(
            audit["angle"],
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
    )

    assert audit["applied"] is True
    delta = np.abs(corrected.astype(np.int16) - expected.astype(np.int16))
    assert int(delta.max()) <= 1
    assert int(np.count_nonzero(delta)) <= 12


def test_core_composes_pixel_deskew_after_rotation_and_crop():
    rotation_and_crop = [[0.0, -1.0, 800.0], [1.0, 0.0, -120.0], [0.0, 0.0, 1.0]]
    pixel_deskew = [[1.0, 0.0, 8.0], [0.0, 1.0, -6.0], [0.0, 0.0, 1.0]]

    composed = extractor_module._compose_pixel_affine(pixel_deskew, rotation_and_crop, zoom=2.0)

    assert np.asarray(composed) == pytest.approx(
        np.asarray([[0.0, -1.0, 804.0], [1.0, 0.0, -123.0], [0.0, 0.0, 1.0]])
    )


def test_core_transform_records_hough_deskew_without_parse_result_change():
    deskew = {
        "method": "hough_lines_p_v1",
        "applied": True,
        "angle": -0.85,
        "reason": "applied",
        "support_line_count": 12,
    }

    transform = extractor_module._logical_page_transform(
        source_page_number=2,
        matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inverse_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_crop_bbox=(0.0, 0.0, 420.0, 594.0),
        rotation=270,
        split_kind="two_page_spread",
        segment_index=1,
        confidence=0.98,
        width=420.0,
        height=594.0,
        deskew=deskew,
    )

    assert transform["deskew_angle"] == -0.85
    assert transform["deskew_applied"] is True
    assert transform["decomposition"]["deskew_method"] == "hough_lines_p_v1"
    assert transform["decomposition"]["deskew_support_line_count"] == 12


def test_coordinate_roundtrip_uses_source_quad_for_fine_rotation():
    from docmirror.evidence import plane as plane_module

    point_matrix = [[0.9998, 0.01745, -4.0], [-0.01745, 0.9998, 6.0], [0.0, 0.0, 1.0]]
    inverse = extractor_module.invert_matrix(point_matrix)
    page = SimpleNamespace(
        page_number=1,
        page_id="page:0001",
        coordinate_transform={
            "matrix": point_matrix,
            "inverse_matrix": inverse,
            "source_crop_bbox": [-20.0, -20.0, 600.0, 800.0],
            "deskew_applied": True,
        },
    )
    atom = EvidenceAtom(id="ocr:1", page_id="page:0001", bbox=[50.0, 80.0, 450.0, 100.0])

    plane_module._attach_page_coordinates(atom, page)
    gates = build_udtr_quality_gates(pages=[page], regions=[], blocks=[], evidence_atoms=[atom])
    gate = next(item for item in gates if item["id"] == "gate:coordinate_roundtrip")

    assert len(atom.metadata["source_quad"]) == 4
    assert gate["status"] == "pass"
    assert gate["details"]["max_roundtrip_error"] <= 0.001


def test_core_split_path_deskews_each_slice_before_final_ocr(monkeypatch, tmp_path):
    import sys

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    slices = [
        SimpleNamespace(
            image=image[:, :100],
            width=50.0,
            height=50.0,
            crop_bbox_oriented=(0.0, 0.0, 50.0, 50.0),
            source_crop_bbox=(0.0, 0.0, 50.0, 50.0),
            source_to_logical=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            logical_to_source=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            selected_rotation=0,
            segment_index=0,
            split_confidence=0.99,
        ),
        SimpleNamespace(
            image=image[:, 100:],
            width=50.0,
            height=50.0,
            crop_bbox_oriented=(50.0, 0.0, 100.0, 50.0),
            source_crop_bbox=(50.0, 0.0, 100.0, 50.0),
            source_to_logical=[[1.0, 0.0, -50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            logical_to_source=[[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            selected_rotation=0,
            segment_index=1,
            split_confidence=0.99,
        ),
    ]
    corrected = [np.full((104, 102, 3), 10, dtype=np.uint8), np.full((104, 102, 3), 20, dtype=np.uint8)]
    order = []

    class FakePixmap:
        samples = image.tobytes()
        height, width, n = image.shape

    class FakePage:
        def get_pixmap(self, **_kwargs):
            return FakePixmap()

    class FakeDoc:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return FakePage()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_fitz = SimpleNamespace(open=lambda _path: FakeDoc(), Matrix=lambda *_args: object())
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    monkeypatch.setattr(extractor_module, "split_or_passthrough", lambda *_args, **_kwargs: slices)
    monkeypatch.setattr(extractor_module, "_select_ocr_orientation", lambda *_args, **_kwargs: ([], 0, image, 100.0, 50.0, {"score": 0.0}))

    deskew_calls = 0

    def deskew(supplied):
        nonlocal deskew_calls
        order.append(("deskew", supplied))
        result = corrected[deskew_calls]
        deskew_calls += 1
        return result, {
            "method": "hough_lines_p_v1",
            "applied": True,
            "angle": -1.0,
            "reason": "applied",
            "support_line_count": 8,
            "forward_matrix": [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            "inverse_matrix": [[1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
        }

    monkeypatch.setattr("docmirror.layout.normalization.hough_deskew_image", deskew)

    class Engine:
        def detect_image_words(self, supplied):
            order.append(("ocr", supplied))
            return [(1, 2, 20, 12, "value", None, None, None, 0.9)]

    monkeypatch.setattr("docmirror.ocr.vision.rapidocr_engine.get_ocr_engine", lambda: Engine())

    pages = extractor_module._ocr_logical_pages_for_pdf_page(
        tmp_path / "source.pdf",
        0,
        1,
        logical_start=1,
        source_width=100.0,
        source_height=50.0,
        decision=extractor_module.PageSplitDecision(should_split=True, confidence=0.99),
        page_split_mode="force",
    )

    assert [item[0] for item in order] == ["deskew", "ocr", "deskew", "ocr"]
    assert order[1][1] is corrected[0]
    assert order[3][1] is corrected[1]
    assert [page.width for page in pages] == [51.0, 51.0]
    assert [page.height for page in pages] == [52.0, 52.0]
    assert all(page.coordinate_transform["deskew_applied"] is True for page in pages)
    assert pages[0].image is corrected[0]
    assert pages[1].image is corrected[1]


def test_core_nonsplit_path_reocrs_only_after_applied_deskew(monkeypatch, tmp_path):
    import sys

    source = np.zeros((100, 200, 3), dtype=np.uint8)
    oriented = np.full_like(source, 5)
    corrected = np.full((104, 202, 3), 9, dtype=np.uint8)
    observed = []

    class FakePixmap:
        samples = source.tobytes()
        height, width, n = source.shape

    class FakePage:
        def get_pixmap(self, **_kwargs):
            return FakePixmap()

    class FakeDoc:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return FakePage()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda _path: FakeDoc(), Matrix=lambda *_args: object()),
    )

    class Engine:
        def detect_image_words(self, supplied):
            observed.append(supplied)
            return [(1, 2, 20, 12, "value", None, None, None, 0.9)]

    engine = Engine()
    monkeypatch.setattr("docmirror.ocr.vision.rapidocr_engine.get_ocr_engine", lambda: engine)
    monkeypatch.setattr(
        extractor_module,
        "_select_ocr_orientation",
        lambda *_args, **_kwargs: (
            [(1, 2, 20, 12, "preliminary", None, None, None, 0.8)],
            90,
            oriented,
            50.0,
            100.0,
            {"score": 1.0},
        ),
    )
    monkeypatch.setattr(
        "docmirror.layout.normalization.hough_deskew_image",
        lambda supplied: (
            corrected,
            {
                "method": "hough_lines_p_v1",
                "applied": True,
                "angle": -1.0,
                "reason": "applied",
                "support_line_count": 8,
                "forward_matrix": [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
                "inverse_matrix": [[1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            },
        ),
    )

    blocks, image, deskew = extractor_module._ocr_blocks_for_pdf_page(tmp_path / "source.pdf", 0, 1)

    assert observed == [corrected]
    assert image is corrected
    assert blocks[0].raw_content == "value"
    assert deskew["normalized_page_width"] == 101.0
    assert deskew["normalized_page_height"] == 52.0
