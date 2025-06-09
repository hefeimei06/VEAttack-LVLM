import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_owl2.evaluate.attacks.pgd_veattack import pgd_veattack

import numpy as np
import open_clip
import torch.nn.functional as F
from torchvision.transforms import transforms

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize
        # if args['vision_encoder_pretrained'] != 'openai':
        #     self.model.load_state_dict(torch.load(args['vision_encoder_pretrained'], map_location='cpu'))

        self.model.load_state_dict(torch.load('/home/VEAttack/ckpt/tecoa_eps_4.pt', map_location='cpu'))
    def forward(self, vision, output_normalize, tokens=False):
        vision = torch.nn.functional.interpolate(vision, size=(224, 224), mode='bilinear', align_corners=False)
        if not tokens:
            feature = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
            return feature
        else:
            self.model.output_tokens = True
            feature, calculated_tokens = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
                calculated_tokens = F.normalize(calculated_tokens, dim=-1)
            return feature, calculated_tokens

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


ds_collections = {
    'flickr': {
        'train': 'data/flickr/flickr30k_karpathy_test.json',
        'test': 'data/flickr/flickr30k_karpathy_test.json',
    },
    'coco':{
        'train': '/home/VEAttack/mPLUG-Owl/mPLUG-Owl2/coco_caption_annotations.json',
        'test': '/home/VEAttack/mPLUG-Owl/mPLUG-Owl2/coco_caption_annotations.json',
    }
}


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, image_processor, few_shot=0):

        data = json.load(open(test))
        print(f"Top-level type: {type(data)}")

        if isinstance(data, list):
            print(f"Length of list: {len(data)}")
            print("First item structure:")
            print(json.dumps(data[0], indent=4))
    
        elif isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            for key in data:
                print(f"Key '{key}' -> {type(data[key])}")
                if isinstance(data[key], (list, dict)):
                    print(f"First item/element in key '{key}':")
                    if isinstance(data[key], list) and len(data[key]) > 0:
                        print(json.dumps(data[key][0], indent=4))
                    elif isinstance(data[key], dict):
                        print(json.dumps(data[key], indent=4))
        
        self.images = json.load(open(test))['images']
        self.prompt = prompt
        self.image_processor = image_processor

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = json.load(open(train))['annotations']
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id, image_path = self.images[idx]['id'], self.images[idx]['image']

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], self.image_processor)

        return {
            'image_id': image_id,
            'image_tensor': image_tensor,
            'input_text': self.prompt.format(image_path)
        }


def collate_fn(inputs, tokenizer):
    image_ids = [_['image_id'] for _ in inputs]
    image_tensor = [_['image_tensor'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]

    input_ids = []
    for input_text in input_texts:
        input_ids.append(tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').tolist())
    input_tokens_max_length = max([len(x) for x in input_ids])
    pad_token_id = tokenizer.pad_token_id

    input_ids = [([pad_token_id] * (input_tokens_max_length - len(_)) + _) for _ in input_ids] # pad in the left
    input_ids = torch.LongTensor(input_ids)
    attention_mask = 1 - input_ids.eq(pad_token_id).long()

    image_tensor = torch.cat(image_tensor, dim=0)
    return image_ids, image_tensor, input_ids, attention_mask


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    prompt = 'USER: <|image|>Provide a one-sentence caption for the provided image. ASSISTANT: '

    model_path = args.checkpoint
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device_map={"":f"cuda:{os.getenv('LOCAL_RANK', '0')}"}, device="cuda")
    tokenizer.padding_side = 'left'
    if not hasattr(tokenizer, 'pad_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    random.seed(args.seed)
    dataset = CaptionDataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        image_processor=image_processor,
        few_shot=args.few_shot,
    )
    np.random.seed(44)
    random_indices = np.random.choice(len(dataset), 500, replace=False)
    dataset = torch.utils.data.Subset(dataset, random_indices)
    coco_karpathy_test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    image_ids = []
    captions = []

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
    args.eps = 16 / 255

    for _, (ids, image_tensor, input_ids, attention_mask) in enumerate(tqdm(coco_karpathy_test_loader)):

        mean_tensor = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
        image_tensor = image_tensor * std_tensor + mean_tensor

        args.output_normalize = False
        image_tensor = image_tensor.to(model.device)
        with torch.no_grad():
            embedding_orig, tokens_orig = clip_model_vision(vision=image_tensor,
                                                            output_normalize=args.output_normalize, tokens=True)

       loss_inner_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, 'mean')
        args.norm = 'linf'
        args.iterations_adv = 100  # 10, 50
        args.stepsize_adv = 4 / 255  # 1 / 255
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

        image_tensor = transforms.Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
            )(image_tensor)
   

        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            images=image_tensor.to(dtype=model.dtype).cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=60,
            min_new_tokens=8,
            length_penalty=0,
            num_return_sequences=1,
            use_cache=True,
        )
        image_ids.extend(ids)
        captions.extend([
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ])
        print(captions[-len(pred):])

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_captions = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_captions, captions)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_captions = [
        _ for _ in itertools.chain.from_iterable(merged_captions)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for image_id, caption in zip(merged_ids, merged_captions):
            results.append({
                'image_id': int(image_id),
                'caption': caption,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

        coco = COCO(ds_collections[args.dataset]['test'])
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()

        print(coco_eval.eval.items())
    torch.distributed.barrier()