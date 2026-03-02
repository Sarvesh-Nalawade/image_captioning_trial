
from paddleocr import PaddleOCR
import json

# Use default English OCR model
ocr = PaddleOCR(lang='en')

# Path to your test image (update as needed)
image_path = '/Users/sarvesh/MyMacbook/Projects/AlgoAnalytics/image_captioning_trial/sample_images/sample_img_1.png'


def _clean_text_list(candidates):
	seen = set()
	cleaned = []
	for item in candidates:
		if not isinstance(item, str):
			continue
		text = item.strip()
		if not text or text in seen:
			continue
		seen.add(text)
		cleaned.append(text)
	return cleaned


def extract_text_from_result(obj):
	text_candidates = []

	def walk(node):
		if isinstance(node, dict):
			for key, value in node.items():
				lower_key = str(key).lower()

				if lower_key in {'text', 'texts', 'rec_text', 'rec_texts', 'ocr_text'}:
					if isinstance(value, str):
						text_candidates.append(value)
					elif isinstance(value, list):
						for entry in value:
							if isinstance(entry, str):
								text_candidates.append(entry)
							else:
								walk(entry)
					else:
						walk(value)
				else:
					walk(value)

		elif isinstance(node, (list, tuple)):
			# Handle classic OCR tuple format: [box, (text, score)]
			if (
				len(node) >= 2
				and isinstance(node[1], (list, tuple))
				and len(node[1]) >= 1
				and isinstance(node[1][0], str)
			):
				text_candidates.append(node[1][0])

			for item in node:
				walk(item)

	walk(obj)
	return _clean_text_list(text_candidates)


# First try predict() output
result = ocr.predict(image_path)
extracted_texts = extract_text_from_result(result)

# Fallback to classic ocr() if predict() parsed no text
if not extracted_texts:
	fallback_result = ocr.ocr(image_path, cls=False)
	extracted_texts = extract_text_from_result(fallback_result)

output_path = 'ocr_output.json'
with open(output_path, 'w', encoding='utf-8') as f:
	json.dump(extracted_texts, f, indent=2, ensure_ascii=False)

print(f"OCR extracted {len(extracted_texts)} text line(s), saved to {output_path}")