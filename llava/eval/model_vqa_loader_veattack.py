import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from open_flamingo.eval.models.of_eval_model_adv import EvalModelAdv
from open_flamingo.eval.vqa_metric import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
)
from PIL import Image
import math

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import open_clip
import torch.nn.functional as F

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, vision, output_normalize, tokens=False):
        if not tokens:
            feature = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
            return feature
        else:
            self.model.output_tokens = True
            feature, tokens = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
                tokens = F.normalize(tokens, dim=-1)
            return feature, tokens



class ComputeLossWrapper:
    def __init__(self, embedding_orig, tokens_orig, reduction='mean'):
        self.embedding_orig = embedding_orig
        self.reduction = reduction
        self.tokens_orig = tokens_orig

    def __call__(self, embedding, tokens):
        return compute_loss(embedding=embedding, embedding_orig=self.embedding_orig,
                            tokens=tokens, tokens_orig=self.tokens_orig, reduction=self.reduction)


def compute_loss(embedding, embedding_orig, tokens, tokens_orig, reduction='mean'):

    loss = cosine_similarity_loss(out=tokens, targets=tokens_orig, reduction=reduction)

    return loss


def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch


def kl_divergence(out, targets, reduction='none', eps=1e-10):
    """
    Stable KL divergence computation with eps clipping.
    Args:
        out: shape (batch_size, seq_len, hidden_dim), e.g., (1, 256, 1024)
        targets: same shape as `out`
        reduction: 'none' | 'mean'
        eps: small value to avoid log(0)
    Returns:
        KL divergence (possibly averaged if reduction='mean')
    """
    # Stable softmax + log-softmax
    out_logprobs = F.log_softmax(out, dim=-1)  # log(p)
    targets_probs = F.softmax(targets, dim=-1)  # q

    # Clip probabilities to avoid log(0)
    targets_probs = torch.clamp(targets_probs, min=eps, max=1.0)

    # Compute KL(p||q) = sum(p * (log(p) - log(q)))
    kl_per_token = (targets_probs * (torch.log(targets_probs) - out_logprobs)).sum(dim=-1)

    if reduction == 'mean':
        return kl_per_token.mean()
    else:
        return kl_per_token

def cosine_similarity_loss(out, targets, reduction='none', eps=1e-8):
    """
    Compute cosine similarity loss (1 - cosine_similarity).
    Args:
        out: shape (batch_size, seq_len, hidden_dim), e.g., (1, 256, 1024)
        targets: same shape as `out`
        reduction: 'none' | 'mean'
        eps: small value to avoid division by zero
    Returns:
        - If reduction='none': shape (batch_size, seq_len), e.g., (1, 256)
        - If reduction='mean': scalar
    """
    # Normalize the embeddings to unit vectors
    out_norm = F.normalize(out, p=2, dim=-1)  # (1, 256, 1024)
    targets_norm = F.normalize(targets, p=2, dim=-1)  # (1, 256, 1024)

    # Compute cosine similarity: sum(out * targets) / (||out|| * ||targets||)
    cosine_sim = (out_norm * targets_norm).sum(dim=-1)  # (1, 256)

    # Loss = 1 - cosine_similarity (to minimize)
    loss = 1.0 - cosine_sim  # (1, 256)

    if reduction == 'mean':
        return loss.mean()  # average over batch and seq_len
    else:
        return loss  # (1, 256)

def decosine_similarity_loss(out, targets, reduction='none', eps=1e-8):
    """
    Compute cosine similarity loss (1 - cosine_similarity).
    Args:
        out: shape (batch_size, seq_len, hidden_dim), e.g., (1, 256, 1024)
        targets: same shape as `out`
        reduction: 'none' | 'mean'
        eps: small value to avoid division by zero
    Returns:
        - If reduction='none': shape (batch_size, seq_len), e.g., (1, 256)
        - If reduction='mean': scalar
    """
    # Normalize the embeddings to unit vectors
    out_norm = F.normalize(out, p=2, dim=-1)  # (1, 256, 1024)
    targets_norm = F.normalize(targets, p=2, dim=-1)  # (1, 256, 1024)

    # Compute cosine similarity: sum(out * targets) / (||out|| * ||targets||)
    cosine_sim = (out_norm * targets_norm).sum(dim=-1)  # (1, 256)

    # Loss = 1 - cosine_similarity (to minimize)
    loss = cosine_sim  # (1, 256)

    if reduction == 'mean':
        return loss.mean()  # average over batch and seq_len
    else:
        return loss  # (1, 256)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_of_args(pretrained_rob_path=None):
    model_args = {}
    model_args['vision_encoder_pretrained'] = pretrained_rob_path
    model_args['vision_encoder_path'] =  'ViT-L-14'
    model_args['lm_path'] = 'anas-awadalla/mpt-7b'
    model_args['lm_tokenizer_path'] = 'anas-awadalla/mpt-7b'
    model_args['checkpoint_path'] = './ckpt/checkpoint.pt'
    # model_args['device'] = 'cuda'
    model_args['cross_attn_every_n_layers'] =  4 
    model_args['precision'] = 'float16'

    return model_args

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, model='LLAVA'):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model = model

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        
        if self.model == 'LLAVA':
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if self.model == 'LLAVA':
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        else:
            image = Image.open(os.path.join(self.image_folder, image_file))
            # image.load()
            transform = transforms.Compose([
            transforms.ToTensor()
            ])
            image_tensor = transform(image) #.squeeze(0) #.load()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, index

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, model='LLAVA'):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, model)
    np.random.seed(42)
    random_indices = np.random.choice(len(dataset), 500, replace=False)
    dataset = torch.utils.data.Subset(dataset, random_indices)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # collate_fn=custom_collate_fn,
        shuffle=False
    )
    # data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def inverse_normalize(image_normalized, image_processor):
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1).to(image_normalized.device)
    std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1).to(image_normalized.device)
    return torch.clamp(image_normalized * std + mean, 0, 1)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.pretrained_rob_path == 'None':
        args.pretrained_rob_path = None
    print(f"Model at: {args.pretrained_rob_path}")
    print(f"Need to load llava")
    
    if args.eval_model == 'LLAVA':
        model, image_processor, tokenizer, context_len = load_pretrained_model(model_path, args.model_base, model_name, pretrained_rob_path=args.pretrained_rob_path, dtype='float16')
    else:
        _, image_processor, tokenizer, context_len = load_pretrained_model(model_path, args.model_base, model_name, pretrained_rob_path=args.pretrained_rob_path, dtype='float16')
        model_args = get_of_args(args.pretrained_rob_path)
        eval_model = EvalModelAdv(model_args, adversarial=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        device_id = 0
        eval_model.set_device(device_id)
        # model.config = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor if args.eval_model == 'LLAVA' else None, model.config if args.eval_model == 'LLAVA' else None, model=args.eval_model)


    args.clip_model_name = 'ViT-L-14'
    clip_model_orig, _, image_processor_clip = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )

    # Remove the Normalize transform by creating a new Compose object
    normalize = image_processor_clip.transforms[-1]
    del image_processor_clip
    clip_model_orig.cpu()
    clip_model_vision = ClipVisionModel(model=clip_model_orig.visual, normalize=normalize)
    clip_model_vision.cuda()
    clip_model_orig.cuda()
    args.eps = args.eps / 255

    for (input_ids, image_tensor, index) in tqdm(data_loader, total=len(data_loader)):
        line = questions[index]
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        
        if args.eval_model == 'LLAVA':

            image_tensor = inverse_normalize(image_tensor, image_processor)

            if args.attack == 'veattack':
                args.output_normalize = False
                image_tensor = image_tensor.to(device='cuda', non_blocking=True)
                with torch.no_grad():
                    embedding_orig, tokens_orig = clip_model_vision(vision=image_tensor,
                                                                    output_normalize=args.output_normalize,
                                                                    tokens=True)

                loss_inner_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, 'mean')
                args.norm = 'linf'
                args.eps = args.eps
                args.iterations_adv = 100
                args.stepsize_adv = 1 / 255
                from vlm_eval.attacks.veattack import pgd_veattack
                image_tensor = pgd_veattack(
                    forward=clip_model_vision,
                    loss_fn=loss_inner_wrapper,
                    data_clean=image_tensor,
                    norm=args.norm,
                    eps=args.eps,
                    iterations=args.iterations_adv,
                    stepsize=args.stepsize_adv,
                    output_normalize=args.output_normalize,
                    perturbation=torch.zeros_like(image_tensor).uniform_(-args.eps, args.eps).requires_grad_(True),
                    mode='max',
                    verbose=False
                )


            normalizer = transforms.Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            )
            image_tensor = normalizer(image_tensor)



            stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
            input_ids = input_ids.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            predictions = outputs.strip()
        
        else:
            transs = transforms.ToPILImage()
            ims = []
            ims.append(transs(image_tensor.squeeze()))
            image_tensor = []
            image_tensor.append(ims)
            batch_images = eval_model._prepare_images(image_tensor)
            
            batch_images = batch_images.squeeze(1).squeeze(1)

            if args.attack == 'veattack':
                args.output_normalize = False
                batch_images = batch_images.to(device='cuda', non_blocking=True)
                with torch.no_grad():
                    embedding_orig, tokens_orig = clip_model_vision(vision=batch_images,
                                                                    output_normalize=args.output_normalize,
                                                                    tokens=True)

                
                loss_inner_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, 'mean')
                args.norm = 'linf'
                args.eps = args.eps
                args.iterations_adv = 100
                args.stepsize_adv = 1 / 255
                from vlm_eval.attacks.veattack import pgd_veattack
                batch_images = pgd_veattack(
                    forward=clip_model_vision,
                    loss_fn=loss_inner_wrapper,
                    data_clean=batch_images,
                    norm=args.norm,
                    eps=args.eps,
                    iterations=args.iterations_adv,
                    stepsize=args.stepsize_adv,
                    output_normalize=args.output_normalize,
                    perturbation=torch.zeros_like(batch_images).uniform_(-args.eps, args.eps).requires_grad_(True),
                    mode='max',
                    verbose=False
                )
            
            batch_images = batch_images.unsqueeze(1).unsqueeze(1)
            
            batch_text = []
            # yes_no = random.choice(['yes', 'no'])
            # add_str_1 = 'Is there some object in the image?'
            # add_str_2 = 'Is the image taken during day time?'
            # context_text = f"Question:{add_str_1} answer:{yes_no}<|endofchunk|>"
            # context_text += f"Question:{add_str_2} answer:{yes_no}<|endofchunk|>"
            context_text = f"Question:{cur_prompt} answer:"
            # Keep the text but remove the image tags for the zero-shot case
            # if num_shots == 0:
            #     context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=cur_prompt)
            )
            # print(cur_prompt)
            # batch_text.append(cur_prompt)
            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                min_generation_length=0,
                max_generation_length=5,
                num_beams=3,
                length_penalty=1.0,
            )
            dataset_name = 'coco'
            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs) #.strip()
            predictions = []
            for new_prediction, sample_id in zip(new_predictions, cur_prompt):
                predictions.append(new_prediction)
                # outputs = outputs.strip()
            predictions = predictions[0].strip()
            # print(predictions)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": predictions,
                                   "answer_id": ans_id,
                                   "model_id": model_name if args.eval_model == 'LLAVA' else args.eval_model,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--pretrained_rob_path", type=str, default='openai', help='Pass None, openai or path-to-rob-ckpt')
        # "/data/naman_deep_singh/project_multimodal/clip-finetune/sbatch/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-eps4-3adv-lr1e-4-wd-1e-3_f8o0v/checkpoints/final.pt")
        # /home/nsingh/project_multimodal/models/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-eps4-3adv-lr1e-4-wd-1e-3_f8o0v/checkpoints/final.pt
    parser.add_argument("--eval-model", type=str, default='LLAVA')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--attack", type=str, default="none")
    parser.add_argument("--eps", type=int, default=2)
    args = parser.parse_args()
    eval_model(args)
