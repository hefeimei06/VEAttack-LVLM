import json

def remove_duplicate_image_ids(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    seen = set()
    unique_results = []
    for item in results:
        img_id = item['image_id']
        if img_id not in seen:
            seen.add(img_id)
            unique_results.append(item)
        else:
            print(f"Detected duplicate: image_id={img_id}, skipped.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, ensure_ascii=False, indent=2)

    print(f"Save the new file to: {output_file}")

input_path = '/home/VEAttack/MiniGPT-4/caption_clean_results.json'
output_path = '/home/VEAttack/MiniGPT-4/caption_single_results.json'

remove_duplicate_image_ids(input_path, output_path)
