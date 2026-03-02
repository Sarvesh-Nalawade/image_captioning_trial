import json
from transformers import pipeline

JSON_PATH = 'sample_files/image_text_mapping_summarized.json'
OUTPUT_JSON_PATH = 'sample_files/image_captioned_local_t5.json'

# Use t5-small for lightweight summarization/captioning
summarizer = pipeline('summarization', model='t5-small')

def generate_caption(summary):
    if not summary.strip():
        return ''
    prompt = "summarize: " + summary.strip()
    try:
        result = summarizer(prompt, max_length=15, min_length=5, do_sample=False)
        return result[0]['summary_text'].strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return ''

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for item in data:
    summary = item.get('summary', '')
    item['caption'] = generate_caption(summary)

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Image captions saved to {OUTPUT_JSON_PATH}")
