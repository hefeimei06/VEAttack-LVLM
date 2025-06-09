#!/bin/bash
# LLaVA evaluation script

python -m vlm_eval.run_evaluation_veattack \
--eval_coco \
--attack veattack --eps 4 --steps 100 --mask_out none \
--vision_encoder_pretrained openai \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path out_llava_13b/eval_llava_4_1_100_coco \
--model_path liuhaotian/llava-v1.5-13b \
--coco_train_image_dir_path /home/datasets/coco2014/train2014 \
--coco_val_image_dir_path /home/datasets/coco2014/val2014 \
--coco_karpathy_json_path /home/datasets/coco2014/annotations/karpathy_coco.json \
--coco_annotations_json_path /home/datasets/coco2014/annotations/captions_val2014.json \
--flickr_image_dir_path /home/datasets/flickr30k/flickr30k-images \
--flickr_karpathy_json_path /home/datasets/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /home/datasets/flickr30k/dataset_flickr30k_coco_style.json \
--vqav2_train_image_dir_path /home/datasets/coco2014/train2014 \
--vqav2_train_questions_json_path /home/datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /home/datasets/VQAv2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /home/datasets/coco2014/val2014 \
--vqav2_test_questions_json_path /home/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /home/datasets/VQAv2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /home/datasets/textvqa/train_images \
--textvqa_train_questions_json_path /home/datasets/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /home/datasets/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /home/datasets/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /home/datasets/textvqa/val_annotations_vqa_format.json
