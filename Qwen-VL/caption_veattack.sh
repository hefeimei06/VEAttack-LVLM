ds="coco"
checkpoint=/home/.cache/modelscope/hub/models/qwen/Qwen-VL
python3 -m torch.distributed.launch --use-env \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
    /home/VEAttack/Qwen-VL/eval_mm/evaluate_caption_veattack.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 1 \
    --num-workers 1