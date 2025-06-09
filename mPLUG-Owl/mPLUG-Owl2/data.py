import json
import os
from tqdm import tqdm

captions_file = '/home/datasets/coco2014/annotations/karpathy_coco.json'
image_root  = '/home/datasets/coco2014/val2014'
output_file = 'coco_caption_annotations.json'

with open(captions_file, 'r') as f:
    coco_data = json.load(f)

image_id_to_filename = {
    image['imgid']: image['filename']
    for image in coco_data['images']
}
output_data = {
    'annotations': [],
    'images': []
}

seen_image_ids = set()

for ann in coco_data['images']:
    image_id = ann['cocoid']
    caption_id = ann['imgid']
    caption = ann['sentences']["raw"]
    
    file_name = image_id_to_filename.get(image_id)
    if not file_name:
        continue  

    image_path = os.path.join(image_root, file_name)

    output_data['annotations'].append({
        'image_id': image_id,
        'id': caption_id,
        'image': image_path,
        'caption': caption
    })

    if image_id not in seen_image_ids:
        output_data['images'].append({
            'id': image_id,
            'image': image_path
        })
        seen_image_ids.add(image_id)

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"New file saved in：{output_file}")