export PYTHONPATH=$PYTHONPATH:/home/VEAttack/MiniGPT-4

python '/home/VEAttack/MiniGPT-4/eval_scripts/eval_caption_veattack.py'  --cfg-path /home/VEAttack/MiniGPT-4/eval_configs/minigpt4_eval.yaml --dataset cococaption

python '/home/VEAttack/MiniGPT-4/clean.py'

python '/home/VEAttack/MiniGPT-4/single.py'

python '/home/VEAttack/MiniGPT-4/coco_metric.py'