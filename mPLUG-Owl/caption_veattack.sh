ds="coco"
checkpoint=/home/VEAttack/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b
python -m torch.distributed.launch --use-env \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
    /home/VEAttack/mPLUG-Owl/mPLUG-Owl2/mplug_owl2/evaluate/evaluate_caption_veattack.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 1 \
    --num-workers 1