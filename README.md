# Image Captioning Trial

A Python playground for experimenting with image captioning pipelines on medical PDFs/PPTX/images using a mix of:

- **LLMs** (OpenAI GPT, Gemini Flash Lite)
- **Vision captioning** (BLIP)
- **OCR** (PaddleOCR)
- **Local summarization** (`t5-small`)

The repo contains multiple independent scripts plus one end-to-end pipeline (`final_captioning.py`).

---

## Repository contents

### Core scripts

- `extract_images_and_text.py`  
  Extracts images + page text from a PDF (PyMuPDF) into:
  - `sample_files/extracted_images/`
  - `sample_files/image_text_mapping.json`

- `summarize_image_text.py`  
  Uses OpenAI (`gpt-4o-mini`) to summarize each page’s text into `summary`.

- `image_caption_images.py`  
  Uses OpenAI (`gpt-4o-mini`) to generate short captions from summaries.

- `image_caption_local.py`  
  Local alternative: uses `t5-small` summarization pipeline to create short captions.

- `image_caption_blip.py`  
  Adds BLIP-generated image captions (`blip_caption`) for each image.

- `image_caption_local_t5_blip.py`  
  BLIP step applied after `image_caption_local.py` output.

- `gemini_flash_lite_captioning.py`  
  CLI script that captions a single image or image directory with Gemini Flash Lite and writes JSON output.

- `final_captioning.py`  
  Main unified pipeline:
  - accepts JSON / image / PPTX input
  - summarizes slide/page text (local `t5-small`)
  - generates BLIP caption
  - runs PaddleOCR
  - combines all signals into a retrieval-focused long summary (`new_image_summary`) via OpenAI API (with fallback when unavailable)

- `temp.py`  
  OCR parsing/debug utility for one test image (`sample_images/sample_img_1.png`) that writes `ocr_output.json`.

### Data / assets

- `sample_files/` — sample inputs and produced outputs:
  - PPTX files (`NIV.pptx`, etc.)
  - `final_captioning_output.json`
  - `image_captioned_local_t5_blip.json`
  - `extracted_images_from_pptx/`
- `sample_images/` — test images
- `ocr_output.json` — OCR extraction output from `temp.py`

### Model folder

- `PaddleOCR-VL-1.5/` — local model/tokenizer/config artifacts (large model assets).

### Notes

- `.gitignore` currently ignores `sample_files/`, `sample_images/`, `PaddleOCR-VL-1.5/`, and some generated artifacts.
- `OCR vs Gemini Captioning.pages/.pdf` appear to be analysis docs.

---

## Environment setup

## 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

```bash
pip install \
  pymupdf \
  pillow \
  transformers \
  torch \
  openai \
  python-dotenv \
  google-generativeai \
  paddleocr \
  python-pptx \
  requests
```

> `transformers` + `torch` will download model weights on first run for BLIP and `t5-small`.

## 3) Configure API keys (`.env`)

Create a `.env` in repo root:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
# OR GOOGLE_API_KEY=your_gemini_key
```

---

## Workflows

## A) PDF → summarized text + captions (legacy multi-step path)

1. Extract page images + text:

```bash
python extract_images_and_text.py
```

2. Summarize page text (OpenAI):

```bash
python summarize_image_text.py
```

3. Caption from summaries (choose one):

OpenAI captions:
```bash
python image_caption_images.py
```

Local T5 captions:
```bash
python image_caption_local.py
```

4. Add BLIP image captions:

For OpenAI path:
```bash
python image_caption_blip.py
```

For local path:
```bash
python image_caption_local_t5_blip.py
```

---

## B) Unified pipeline (recommended)

Run end-to-end fusion on default sample PPTX (`sample_files/NIV.pptx`):

```bash
python final_captioning.py
```

Or pass your own input (JSON / image / PPTX):

```bash
python final_captioning.py /path/to/input
```

Output:

- `sample_files/final_captioning_output.json`

Each output item can include:

- `summary` (local text summary)
- `blip_caption`
- `ocr_text` (list)
- `new_image_summary` (rich retrieval-focused description)

---

## C) Gemini Flash Lite captioning (standalone)

Caption one image:

```bash
python gemini_flash_lite_captioning.py \
  --input sample_images/sample_img_1.png \
  --output sample_files/gemini_flash_lite_captions.json
```

Caption all images in a directory:

```bash
python gemini_flash_lite_captioning.py \
  --input sample_images \
  --output sample_files/gemini_flash_lite_captions.json \
  --context "Medical ICU training slides"
```

Optional flags:

- `--model` (default: `gemini-flash-lite-latest`)
- `--context` extra prompt context

---

## Output schema (typical)

```json
{
  "image_path": "...",
  "slide": 1,
  "slide_text": "...",
  "summary": "...",
  "blip_caption": "...",
  "ocr_text": ["..."],
  "new_image_summary": "..."
}
```

(Fields vary by script and input type.)

---

## Known caveats

- Some script paths are hardcoded to files under `sample_files/`.
- OpenAI model comments in code may not exactly match model IDs used.
- Large OCR/vision models can be slow on CPU.
- If API keys are missing, `final_captioning.py` falls back to a non-LLM merged summary.

---

## Quick start

If you only want one command to test the full idea:

```bash
python final_captioning.py sample_files/NIV.pptx
```

Then inspect:

- `sample_files/final_captioning_output.json`
