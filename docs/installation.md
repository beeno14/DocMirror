# Installation

## Base Install

```bash
pip install docmirror
```

The base package supports lightweight import, version checks, CLI help, and capability inspection:

```bash
python -c "import docmirror; print(docmirror.__version__)"
docmirror --help
docmirror doctor
```

## Public Extras

Install optional capabilities as needed:

| Extra | Packages | Use Case |
|---|---|---|
| `pdf` | PyMuPDF, pdfplumber | Digital PDF parsing |
| `ocr` | RapidOCR, OpenCV, NumPy | Scanned document OCR |
| `ocr-paddle` | PaddleOCR plus RapidOCR fallback | Hybrid GPU OCR (Paddle runtime installed separately) |
| `layout` | rapid-layout | Optional layout model support |
| `table` | rapid-table | Optional table model support |
| `formula` | rapid-latex-ocr | Formula recognition |
| `office` | python-docx, openpyxl, python-pptx | Word, Excel, PowerPoint |
| `security` | pikepdf | PDF inspection and tamper signals |
| `server` | fastapi, uvicorn, python-multipart | HTTP API server |
| `archive` | rarfile | Archive format support |
| `ai` | openai, google-generativeai | Optional AI/VLM integrations |
| `all` | Public OSS extras | Full public OSS feature set |
| `gpu` | `all` capabilities with PaddleOCR replacing the Rapid-only OCR profile | Reproducible GPU container/runtime profile |
| `dev` | pytest, ruff, mypy, coverage, pre-commit | Development tools |
| `docs` | mkdocs-material, mkdocstrings | Documentation site |

Examples:

```bash
pip install "docmirror[pdf]"
pip install "docmirror[ocr]"
pip install "docmirror[office]"
pip install "docmirror[server]"
pip install "docmirror[all]"
```

## OCR Deployment Profiles

Choose one of two installation profiles:

```bash
# Lightweight CPU deployment: RapidOCR only
python -m pip install "docmirror[ocr]"

# Hybrid CUDA 12.6 deployment: PaddleOCR on GPU, RapidOCR as fallback
python -m pip install "paddlepaddle-gpu==3.3.1" \
  --find-links https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/
python -m pip install "docmirror[ocr-paddle]"
```

The CUDA command is an example for CUDA 12.6; use the PaddlePaddle wheel source
that matches the deployment driver/runtime. `DOCMIRROR_OCR_BACKEND=auto` is the
default and selects PaddleOCR only when a requested GPU is available. Set it to
`rapidocr` to force the lightweight backend, or `paddleocr` for a strict Paddle
deployment that fails instead of silently changing engines. Configure the GPU
with `DOCMIRROR_PADDLE_DEVICE=gpu:0` and the model profile with
`DOCMIRROR_PADDLE_PROFILE=small|server|medium`.

The repository also includes a locked CUDA 12.6 container profile. It requires
Docker Compose, a compatible NVIDIA driver, and NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

The first OCR request downloads official model weights. Compose persists the
PaddleX model directory so later container recreations reuse them.

## Commercial Extensions

Enterprise and finance extensions are distributed separately and are not part of the public `docmirror[all]` extra. They may require a private package index and a commercial license.

## Optional: Legacy `.doc` Support

Binary `.doc` files require LibreOffice and the `soffice` command on `PATH`. Without LibreOffice, DocMirror should return a recoverable feature-unavailable error and ask you to use `.docx` or install LibreOffice.

## Requirements

- Python 3.10+
- Linux, macOS, or Windows
