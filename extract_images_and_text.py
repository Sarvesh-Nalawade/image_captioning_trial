import fitz  # PyMuPDF
import os
import json

PDF_PATH = 'sample_files/CVICU Orientation Manual_1.pdf'
OUTPUT_DIR = 'sample_files/extracted_images'
JSON_PATH = 'sample_files/image_text_mapping.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_images_and_text(pdf_path, output_dir, json_path):
    doc = fitz.open(pdf_path)
    image_data = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        # Collect all text from the page
        all_text = ''
        for tblock in blocks:
            if tblock['type'] == 0:  # Text block
                for line in tblock['lines']:
                    for span in line['spans']:
                        all_text += span['text'] + ' '
        for block in blocks:
            if block['type'] == 1:  # Image block
                img = block['image']
                bbox = block['bbox']
                x0, y0, x1, y1 = bbox
                # Save image bytes directly
                img_ext = 'jpg' if img[:4] == b'\xff\xd8\xff\xe0' or img[:4] == b'\xff\xd8\xff\xe1' or img[:4] == b'\xff\xd8\xff\xe2' else 'png' if img[:4] == b'\x89PNG' else 'bin'
                img_name = f"page{page_num+1}_img{blocks.index(block)}.{img_ext}"
                img_path = os.path.join(output_dir, img_name)
                with open(img_path, 'wb') as f:
                    f.write(img)
                image_data.append({
                    'page': page_num+1,
                    'image_path': img_path,
                    'bbox': bbox,
                    'page_text': all_text.strip()
                })
    with open(json_path, 'w') as f:
        json.dump(image_data, f, indent=2)

if __name__ == '__main__':
    extract_images_and_text(PDF_PATH, OUTPUT_DIR, JSON_PATH)
