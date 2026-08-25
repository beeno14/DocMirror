import asyncio
from pathlib import Path
from types import SimpleNamespace

from docmirror.input.entry.options import normalize_parse_policy
from docmirror.input.extraction import extractor as extractor_module
from docmirror.input.extraction.extractor import CoreExtractor
from docmirror.input.extraction.page_splitter import DocumentSpreadPlan, PageSplitDecision
from docmirror.models.entities.domain import Block, PageLayout, TextSpan


def _fake_plane():
    pages = [
        SimpleNamespace(
            page_id="page:0001",
            page_index=0,
            page_number=1,
            width=612.0,
            height=792.0,
            content_mode="text",
        ),
        SimpleNamespace(
            page_id="page:0002",
            page_index=1,
            page_number=2,
            width=612.0,
            height=792.0,
            content_mode="text",
        ),
        SimpleNamespace(
            page_id="page:0003",
            page_index=2,
            page_number=3,
            width=612.0,
            height=792.0,
            content_mode="image",
        ),
    ]
    atoms = [
        SimpleNamespace(id="atom:1", page_id="page:0001", text="page one", bbox=[0, 0, 10, 10]),
        SimpleNamespace(id="atom:2", page_id="page:0002", text="native page two", bbox=[0, 0, 20, 10]),
    ]
    return SimpleNamespace(pages=pages, evidence=SimpleNamespace(text_atoms=atoms))


def _ocr_block(text: str, page_number: int) -> Block:
    block_id = f"ocr:{page_number}"
    return Block(
        block_id=block_id,
        block_type="text",
        spans=(TextSpan(text=text, bbox=(0.0, 0.0, 1.0, 1.0)),),
        bbox=(0.0, 0.0, 1.0, 1.0),
        reading_order=0,
        page=page_number,
        raw_content=text,
        evidence_ids=(block_id,),
    )


def test_content_acquisition_modes_ignore_empty_scanned_pages() -> None:
    native = PageLayout(page_number=1, blocks=(_ocr_block("native ledger", 1),), is_scanned=False)
    blank_scanned = PageLayout(page_number=2, blocks=(), is_scanned=True)

    assert extractor_module._content_acquisition_modes([native, blank_scanned]) == (False, True)


def test_content_acquisition_modes_keep_real_hybrid_content() -> None:
    native = PageLayout(page_number=1, blocks=(_ocr_block("native ledger", 1),), is_scanned=False)
    scanned = PageLayout(page_number=2, blocks=(_ocr_block("OCR ledger", 2),), is_scanned=True)

    assert extractor_module._content_acquisition_modes([native, scanned]) == (True, True)


def test_vnext_extractor_respects_page_selection_and_auto_ocr(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return _fake_plane()

    ocr_calls = []

    def fake_ocr(_file_path, _page_index, page_number, *, start_order=0):
        ocr_calls.append(page_number)
        return [_ocr_block(f"ocr page {page_number}", page_number)]

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    monkeypatch.setattr(extractor_module, "_ocr_blocks_for_pdf_page", fake_ocr)

    policy = normalize_parse_policy(pages="2-3", ocr="auto")
    result = asyncio.run(
        CoreExtractor().extract_parse_result(
            Path("sample.pdf"),
            options={"parse_policy": policy},
        )
    )

    assert [page.page_number for page in result.pages] == [2, 3]
    assert "page one" not in result.full_text
    assert "native page two" in result.full_text
    assert "ocr page 3" in result.full_text
    assert ocr_calls == [3]
    assert result.parser_info.options["selected_pages"] == [2, 3]
    assert result.parser_info.options["ocr_mode"] == "auto"


def test_vnext_extractor_reports_selected_page_completion(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return _fake_plane()

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    events: list[tuple[str, float, str]] = []
    policy = normalize_parse_policy(pages="2-3", ocr="off")

    asyncio.run(
        CoreExtractor().extract_parse_result(
            Path("sample.pdf"),
            options={
                "parse_policy": policy,
                "on_progress": lambda phase, pct, message: events.append((phase, pct, message)),
            },
        )
    )

    page_events = [event for event in events if event[0] == "page_extraction"]
    assert page_events == [
        ("page_extraction", 0.0, "Extracting 0/2 pages..."),
        ("page_extraction", 50.0, "Extracted page 1/2"),
        ("page_extraction", 100.0, "Extracted page 2/2"),
    ]


def test_vnext_extractor_force_ocr_runs_even_with_native_text(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return _fake_plane()

    ocr_calls = []

    def fake_ocr(_file_path, _page_index, page_number, *, start_order=0):
        ocr_calls.append(page_number)
        return [_ocr_block(f"force ocr page {page_number}", page_number)]

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    monkeypatch.setattr(extractor_module, "_ocr_blocks_for_pdf_page", fake_ocr)

    policy = normalize_parse_policy(pages="2", ocr="force")
    result = asyncio.run(
        CoreExtractor().extract_parse_result(
            Path("sample.pdf"),
            options={"parse_policy": policy},
        )
    )

    assert [page.page_number for page in result.pages] == [2]
    assert "native page two" in result.full_text
    assert "force ocr page 2" in result.full_text
    assert ocr_calls == [2]
    assert result.parser_info.options["ocr_mode"] == "force"


def test_vnext_extractor_auto_ocr_replaces_suspicious_native_glyph_mapping(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    plane = _fake_plane()
    suspicious = [
        "客户存款月结单",
        "户名：" + "上" * 12,
        "客户行：" + "平" * 8,
        *("对方户名：" + char * 6 for char in ("备", "往", "咨", "梅", "国", "宋")),
        "温馨提示：单位应与开户银行定期进行对账并及时核对财务信息",
    ]
    plane.evidence.text_atoms = [
        SimpleNamespace(
            id=f"atom:suspicious:{index}",
            page_id="page:0002",
            text=text,
            bbox=[0, index * 10, 200, index * 10 + 8],
        )
        for index, text in enumerate(suspicious)
    ]

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return plane

    def fake_ocr(_file_path, _page_index, page_number, *, start_order=0):
        return [_ocr_block("户名：上海炫酷广告有限公司", page_number)]

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    monkeypatch.setattr(extractor_module, "_ocr_blocks_for_pdf_page", fake_ocr)

    policy = normalize_parse_policy(pages="2", ocr="auto")
    result = asyncio.run(CoreExtractor().extract_parse_result(Path("sample.pdf"), options={"parse_policy": policy}))

    assert "上海炫酷广告有限公司" in result.full_text
    assert "上上上上" not in result.full_text
    assert result.pages[0].page_mode == "scanned_ocr"
    assert result.parser_info.options["native_text_ocr_fallback_pages"] == [2]
    assert "native_text_glyph_mapping_suspected" in result.parser_info.warnings


def test_native_glyph_mapping_detector_ignores_isolated_legitimate_repetition():
    normal_atoms = [
        SimpleNamespace(text="人人都应及时核对账户信息，感谢您的支持"),
        SimpleNamespace(text="哈哈哈只是一个孤立示例，其余内容没有异常重复"),
        SimpleNamespace(text="上海炫酷广告有限公司向供应商支付广告服务费用"),
    ] * 4
    suspicious_atoms = [
        SimpleNamespace(text="户名：" + "上" * 12),
        SimpleNamespace(text="客户行：" + "平" * 8),
        SimpleNamespace(text="对方户名：" + "备" * 7),
        SimpleNamespace(text="摘要：" + "往" * 7),
        SimpleNamespace(text="银行流水交易信息与账户余额核对说明"),
    ] * 3

    assert extractor_module._has_suspicious_native_glyph_mapping(normal_atoms) is False
    assert extractor_module._has_suspicious_native_glyph_mapping(suspicious_atoms) is True
    assert extractor_module._should_ocr_page("off", suspicious_atoms) is False


def test_vnext_extractor_suppresses_text_owned_by_scanned_table(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return _fake_plane()

    def fake_ocr(_file_path, _page_index, page_number, *, start_order=0):
        return [
            Block(
                block_id=f"ocr:{page_number}:title",
                block_type="text",
                raw_content="表格标题",
                page=page_number,
                evidence_ids=(f"ocr:{page_number}:title",),
            ),
            Block(
                block_id=f"ocr:{page_number}:row0",
                block_type="text",
                raw_content="项目",
                page=page_number,
                evidence_ids=(f"ocr:{page_number}:row0",),
            ),
            Block(
                block_id=f"ocr:{page_number}:row1",
                block_type="text",
                raw_content="货币资金",
                page=page_number,
                evidence_ids=(f"ocr:{page_number}:row1",),
            ),
        ]

    def fake_reconstruct(blocks, *, page_number, page_width, page_height, start_order=0):
        _ = (blocks, page_width, page_height)
        return Block(
            block_id=f"scanned_table:p{page_number:04d}:0000",
            block_type="table",
            page=page_number,
            reading_order=start_order,
            raw_content=[["项目"], ["货币资金"]],
            evidence_ids=(f"ocr:{page_number}:row0", f"ocr:{page_number}:row1"),
        )

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    monkeypatch.setattr(extractor_module, "_ocr_blocks_for_pdf_page", fake_ocr)
    monkeypatch.setattr(extractor_module, "reconstruct_scanned_statement_table", fake_reconstruct)

    policy = normalize_parse_policy(pages="3", ocr="force")
    result = asyncio.run(CoreExtractor().extract_parse_result(Path("sample.pdf"), options={"parse_policy": policy}))

    page = result.pages[0]
    assert [text.content for text in page.texts] == ["表格标题"]
    assert len(page.tables) == 1
    assert "货币资金" in result.full_text
    assert all(text.content != "项目" for text in page.texts)


def test_vnext_extractor_expands_one_physical_page_to_two_logical_pages(monkeypatch):
    import docmirror.evidence.plane as evidence_plane_module

    class FakeEvidencePlaneBuilder:
        def build(self, _path):
            return _fake_plane()

    plan = DocumentSpreadPlan(
        mode="auto",
        decisions={3: PageSplitDecision(should_split=True, confidence=0.98, expected_nonblank_segments=2)},
        logical_starts={1: 1, 2: 2, 3: 3},
        logical_page_count=4,
        confidence=0.98,
    )

    def fake_logical_ocr(_file_path, _page_index, source_page_number, **_kwargs):
        transform = {
            "source_page_number": source_page_number,
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        return [
            extractor_module._OcrLogicalPage(3, 3, 420.0, 594.0, (_ocr_block("logical three", 3),), transform),
            extractor_module._OcrLogicalPage(4, 3, 420.0, 594.0, (_ocr_block("logical four", 4),), transform),
        ]

    monkeypatch.setattr(evidence_plane_module, "EvidencePlaneBuilder", FakeEvidencePlaneBuilder)
    monkeypatch.setattr(extractor_module, "_build_pdf_spread_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(extractor_module, "_probe_document_ocr_rotation", lambda *_args, **_kwargs: 270)
    monkeypatch.setattr(extractor_module, "_ocr_logical_pages_for_pdf_page", fake_logical_ocr)

    policy = normalize_parse_policy(pages="3", ocr="auto", page_split="auto")
    result = asyncio.run(CoreExtractor().extract_parse_result(Path("sample.pdf"), options={"parse_policy": policy}))

    assert [page.page_number for page in result.pages] == [3, 4]
    assert [page.source_page_number for page in result.pages] == [3, 3]
    assert all(page.page_mode == "scanned_ocr" for page in result.pages)
    assert result.parser_info.options["source_page_count"] == 3
    assert result.parser_info.options["logical_page_count"] == 4
    assert result.parser_info.options["selected_source_pages"] == [3]


def test_ocr_orientation_metrics_trigger_probe_for_garbage_and_reward_early_title():
    garbage = [(0, 0, 10, 10, "000000000", None, None, None, 0.9) for _ in range(6)]
    good = [
        (0, 0, 10, 10, "所有者权益变动表", None, None, None, 0.9),
        (0, 0, 10, 10, "实收资本", None, None, None, 0.9),
        (0, 0, 10, 10, "250,000,000.00", None, None, None, 0.9),
    ]

    garbage_metrics = extractor_module._ocr_orientation_metrics(garbage)
    good_metrics = extractor_module._ocr_orientation_metrics(good)

    assert extractor_module._needs_orientation_probe(garbage_metrics) is True
    assert good_metrics["early_keywords"] >= 1
    assert good_metrics["score"] > garbage_metrics["score"]
