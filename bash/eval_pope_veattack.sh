#!/bin/bash

baseModel='LLAVA'
# baseModel='openFlamingo'

modelPath=openai
modelPath2=veattack_eps2
modelPath2_2=veattack_eps4


answerFile2="${baseModel}_${modelPath2}"
echo "Will save to the following json: "
echo $answerFile2

python -m llava.eval.model_vqa_loader_veattack \
    --model-path liuhaotian/llava-v1.5-7b \
    --eval-model ${baseModel} \
    --pretrained_rob_path ${modelPath} \
    --question-file ./pope_eval/llava_pope_test.jsonl \
    --image-folder /home/datasets/coco2014/val2014 \
    --answers-file ./pope_eval/${answerFile2}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --attack veattack \
    --eps 2

# pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile2}
python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile2}
python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile2}

python llava/eval/eval_pope_sample.py \
    --model-name $answerFile2 \
    --annotation-dir ./pope_eval/coco_sample/ \
    --question-file ./pope_eval/llava_pope_test_sample.jsonl \
    --result-file ./pope_eval/${answerFile2}_sorted.jsonl


# answerFile2_2="${baseModel}_${modelPath2_2}"
# echo "Will save to the following json: "
# echo $answerFile2_2

# python -m llava.eval.model_vqa_loader_veattack \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --eval-model ${baseModel} \
#     --pretrained_rob_path ${modelPath} \
#     --question-file ./pope_eval/llava_pope_test.jsonl \
#     --image-folder /home/datasets/coco2014/val2014 \
#     --answers-file ./pope_eval/${answerFile2_2}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --attack veattack \
#     --eps 4

# # pope_eval/sort_resule.py, sample_jsonl.py, sample_json.py
# python pope_eval/sort_result.py --result-file ./pope_eval/${answerFile2_2}
# python pope_eval/sample_jsonl.py --result-file ./pope_eval/${answerFile2_2}
# python pope_eval/sample_json.py --result-file ./pope_eval/${answerFile2_2}

# python llava/eval/eval_pope_sample.py \
#     --model-name $answerFile2_2 \
#     --annotation-dir ./pope_eval/coco_sample/ \
#     --question-file ./pope_eval/llava_pope_test_sample.jsonl \
#     --result-file ./pope_eval/${answerFile2_2}_sorted.jsonl

