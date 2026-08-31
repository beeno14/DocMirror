from __future__ import annotations

import threading

import numpy as np

from docmirror.ocr.vision.paddleocr_engine import PaddleOCREngine


class _Result:
    def __init__(self, payload):
        self.json = {"res": payload}


class _Pipeline:
    def __init__(self, results):
        self.results = results
        self.inputs = []

    def predict(self, *, input, **_kwargs):
        self.inputs.append(input)
        return list(self.results)


def _engine_with_pipeline(pipeline) -> PaddleOCREngine:
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.device = "gpu:0"
    engine.profile = "server"
    engine._det_model = "PP-OCRv6_small_det"
    engine._rec_model = "PP-OCRv6_medium_rec"
    engine._lock = threading.RLock()
    engine._pipeline = pipeline
    engine._recognizer = None
    return engine


def test_detection_normalizes_paddle_result_to_canonical_nine_tuple() -> None:
    pipeline = _Pipeline(
        [
            _Result(
                {
                    "rec_texts": ["账户信息", "1720240224"],
                    "rec_scores": np.array([0.97, 0.91]),
                    "rec_boxes": np.array([[10, 20, 110, 40], [20, 50, 120, 70]]),
                }
            )
        ]
    )
    engine = _engine_with_pipeline(pipeline)

    words = engine.detect_image_words(np.zeros((100, 150, 3), dtype=np.uint8))

    assert all(len(word) == 9 for word in words)
    assert words[0][:5] == (10.0, 20.0, 110.0, 40.0, "账户信息")
    assert words[0][8] == 0.97
    # Preserve DocMirror's legacy sequence/date split behavior.
    assert [word[4] for word in words[1:]] == ["17", "20240224"]
    assert [word[7] for word in words] == [0, 1, 2]


def test_detection_accepts_polygon_results() -> None:
    pipeline = _Pipeline(
        [
            _Result(
                {
                    "rec_texts": ["文本"],
                    "rec_scores": [0.88],
                    "rec_polys": [[[1, 2], [11, 2], [11, 9], [1, 9]]],
                }
            )
        ]
    )
    engine = _engine_with_pipeline(pipeline)

    assert engine.detect_image_words(np.zeros((20, 20, 3), dtype=np.uint8))[0][:5] == (
        1.0,
        2.0,
        11.0,
        9.0,
        "文本",
    )


def test_detection_accepts_numpy_text_arrays_without_truth_value_coercion() -> None:
    pipeline = _Pipeline(
        [
            _Result(
                {
                    "rec_texts": np.array(["甲", "乙"]),
                    "rec_scores": np.array([0.9, 0.8]),
                    "rec_boxes": np.array([[1, 2, 10, 9], [12, 2, 20, 9]]),
                }
            )
        ]
    )
    engine = _engine_with_pipeline(pipeline)

    assert [word[4] for word in engine.detect_image_words(np.zeros((30, 30, 3), dtype=np.uint8))] == ["甲", "乙"]


def test_forced_recognition_preserves_exact_input_regions(monkeypatch) -> None:
    pipeline = _Pipeline([])
    recognizer = _Pipeline(
        [
            _Result({"rec_text": "姓名", "rec_score": 0.95}),
            _Result({"rec_text": "低置信", "rec_score": 0.2}),
        ]
    )
    engine = _engine_with_pipeline(pipeline)
    engine._recognizer = recognizer
    monkeypatch.setenv("DOCMIRROR_PADDLE_FORCE_REC_MIN_CONFIDENCE", "0.5")
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    results = engine.force_recognize_regions(image, [(3, 4, 30, 20), (40, 45, 90, 80)])

    assert results == [(3.0, 4.0, 30.0, 20.0, "姓名", 0.95)]


def test_multiscale_merge_removes_overlapping_duplicates() -> None:
    words = [
        (0.0, 0.0, 100.0, 20.0, "A", 0, 0, 0, 0.9),
        (2.0, 1.0, 98.0, 19.0, "A", 0, 0, 1, 0.95),
        (0.0, 30.0, 100.0, 50.0, "B", 0, 1, 2, 0.8),
    ]

    merged = PaddleOCREngine._merge_multiscale(words)

    assert [word[4] for word in merged] == ["A", "B"]
    assert [word[7] for word in merged] == [0, 1]
