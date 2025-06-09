import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config

from pycocoevalcap.eval import COCOEvalCap 

class CaptionEvalDataset(Dataset):
    def __init__(self, annotation_data, vis_processor, img_root):
        self.data = annotation_data
        self.vis_processor = vis_processor
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.img_root, item["image"]))
        image = self.vis_processor(image)
        print(self.vis_processor)
        image_id = item["image_id"]
        return image, image_id

parser = eval_parser()
parser.add_argument("--dataset", type=str, default="cococaption", help="Dataset name")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
caption_cfg = cfg.config["evaluation_datasets"]["cococaption"]
eval_file_path = caption_cfg["eval_file_path"]
img_path = caption_cfg["img_path"]
batch_size = caption_cfg["batch_size"]
max_new_tokens = caption_cfg["max_new_tokens"]

with open(eval_file_path, "r") as f:
    annotation_data = json.load(f)

data = CaptionEvalDataset(annotation_data["annotations"], vis_processor, img_path)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

results = []


for i, (images, image_ids) in enumerate(tqdm(dataloader)):
    if i >= 1000:
        break

    texts = ["<Img><ImageHere></Img> Describe the image in English:"] * len(images)
    # print(images.max())

    images = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )(images)

    answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
    for image_id, caption in zip(image_ids, answers):
        results.append({"image_id": int(image_id), "caption": caption.strip().replace("<unk>", "")})

save_path = cfg.config['run']['save_path']
result_path = os.path.join(save_path, "caption_results.json")
with open(result_path, "w") as f:
    json.dump(results, f)

# from pycocotools.coco import COCO

# annFile = caption_cfg['gt_path'] 
# coco = COCO(annFile)
# cocoRes = coco.loadRes(result_path)
# cocoEval = COCOEvalCap(coco, cocoRes)
# cocoEval.evaluate()

# for metric, score in cocoEval.eval.items():
#     print(f"{metric}: {score:.3f}")
