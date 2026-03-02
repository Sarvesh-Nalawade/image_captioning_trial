import json
import os
from openai import OpenAI
from dotenv import load_dotenv

JSON_PATH = 'sample_files/image_text_mapping_summarized.json'
OUTPUT_JSON_PATH = 'sample_files/image_captioned.json'

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_caption(summary):
    if not summary.strip():
        return ''
    prompt = (
        "Write a short, clear image caption (max 15 words) based only on this summary: '" + summary.strip() + "'"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ''

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for item in data:
    summary = item.get('summary', '')
    item['caption'] = generate_caption(summary)

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Image captions saved to {OUTPUT_JSON_PATH}")
