import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_MODEL = "gemini-flash-lite-latest"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _clean_caption(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _collect_image_paths(input_path: str) -> List[Path]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    image_files = [
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        raise ValueError(f"No images found in directory: {input_path}")

    return sorted(image_files)


def _build_prompt(extra_context: str = "") -> str:
    context_text = extra_context.strip() if isinstance(extra_context, str) else ""

    base_prompt = (
        "Describe the full image in detail for search and retrieval. "
        "Include scene context, important objects, actions, visible text, colors, setting, and overall meaning. "
        "Write a descriptive paragraph."
    )

    if context_text:
        return f"{base_prompt}\nAdditional context: {context_text}"

    return base_prompt


def generate_image_metadata(
    model: genai.GenerativeModel,
    image_path: Path,
    extra_context: str = "",
) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    prompt = _build_prompt(extra_context)

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": mime_type,
                "data": image_bytes,
            },
        ],
        generation_config={"temperature": 0.2},
    )

    caption = _clean_caption(response.text or "")

    return {
        "image_path": str(image_path),
        "caption": caption,
        "model": DEFAULT_MODEL,
    }


def process_images(input_path: str, output_json: str, model_name: str, extra_context: str = "") -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    image_paths = _collect_image_paths(input_path)
    results: List[Dict[str, Any]] = []

    for image_path in image_paths:
        try:
            item = generate_image_metadata(model, image_path, extra_context)
            item["model"] = model_name
            results.append(item)
            print(f"Processed: {image_path}")
        except Exception as error:
            print(f"Failed: {image_path} -> {error}")
            results.append(
                {
                    "image_path": str(image_path),
                    "caption": "",
                    "model": model_name,
                    "error": str(error),
                }
            )

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved output to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval-oriented image captions using Gemini Flash Lite.")
    parser.add_argument("--input", required=True, help="Image file path or directory containing images")
    parser.add_argument(
        "--output",
        default="sample_files/gemini_flash_lite_captions.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name (default: gemini-flash-lite-latest)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional additional context for caption generation",
    )
    args = parser.parse_args()

    process_images(
        input_path=args.input,
        output_json=args.output,
        model_name=args.model,
        extra_context=args.context,
    )


if __name__ == "__main__":
    main()
