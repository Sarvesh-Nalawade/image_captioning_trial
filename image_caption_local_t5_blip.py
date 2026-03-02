import json
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

JSON_PATH = 'sample_files/image_captioned_local_t5.json'
OUTPUT_JSON_PATH = 'sample_files/image_captioned_local_t5_blip.json'

# Load BLIP model and processor (assumes model is already downloaded/cached)
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

def blip_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"BLIP error for {image_path}: {e}")
        return ''

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for item in data:
    image_path = item.get('image_path', '')
    if image_path and os.path.exists(image_path):
        item['blip_caption'] = blip_caption(image_path)
    else:
        item['blip_caption'] = ''

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Final JSON with BLIP captions saved to {OUTPUT_JSON_PATH}")
