import json
import re

def clean_caption(caption):
    return re.split(r'\n|###', caption)[0].strip()

def batch_clean_captions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'caption' in item:
            item['caption'] = clean_caption(item['caption'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"New json has been saved in: {output_file}")

input_path = '/home/VEAttack/MiniGPT-4/caption_results.json'    
output_path = '/home/VEAttack/MiniGPT-4/caption_clean_results.json'  

batch_clean_captions(input_path, output_path)
