import json
import os
from openai import OpenAI
from dotenv import load_dotenv

JSON_PATH = 'sample_files/image_text_mapping.json'
OUTPUT_JSON_PATH = 'sample_files/image_text_mapping_summarized.json'

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_text(text):
    if not text.strip():
        return ''
    prompt = (
        "Summarize the following medical slide/page text in 2-3 sentences, focusing only on the most relevant information for image context.\n" + text
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ''

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for item in data:
    text = item.get('page_text', '')
    item['summary'] = summarize_text(text)

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Summarized JSON saved to {OUTPUT_JSON_PATH}")
