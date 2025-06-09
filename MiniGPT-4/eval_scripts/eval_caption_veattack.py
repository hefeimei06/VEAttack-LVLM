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

import numpy as np
import open_clip
import torch.nn.functional as F
from torchvision.transforms import transforms

def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')


def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf']:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)

def pgd_veattack(
        forward,
        loss_fn,
        data_clean,
        norm,
        eps,
        iterations,
        stepsize,
        output_normalize,
        perturbation=None,
        mode='min',
        momentum=0.9,
        verbose=False
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)
    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            embedding, tokens = forward(data_clean + perturbation,
                                                     output_normalize=output_normalize, tokens=True)
            loss = loss_fn(embedding, tokens)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            gradient = gradient
            if gradient.isnan().any():  #
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                gradient[gradient.isnan()] = 0.
            # normalize
            gradient = normalize_grad(gradient, p=norm)
            # momentum
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)
            # update
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            # project
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean  # clamp to image space
            assert not perturbation.isnan().any()
            assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                data_clean + perturbation
            ) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return data_clean + perturbation.detach()

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize
        # if args['vision_encoder_pretrained'] != 'openai':
        #     self.model.load_state_dict(torch.load(args['vision_encoder_pretrained'], map_location='cpu'))

        self.model.load_state_dict(torch.load('/home/VEAttack/ckpt/fare_eps_4.pt', map_location='cpu'))
    def forward(self, vision, output_normalize, tokens=False):
        # vision = torch.nn.functional.interpolate(vision, size=(224, 224), mode='bilinear', align_corners=False)
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


class CaptionEvalDataset(Dataset):
    def __init__(self, annotation_data, vis_processor, img_root):
        self.data = annotation_data
        self.vis_processor = vis_processor
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.img_root, item["image"]))  # 'image' 应该是相对路径或文件名
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

for i, (images, image_ids) in enumerate(tqdm(dataloader)):
    if i >= 1000:
        break

    texts = ["<Img><ImageHere></Img> Describe the image in English:"] * len(images)
    # print(images.max())

    #------------attack-----------------------------------------
    args.output_normalize = False
    images = images.to(model.device)
    with torch.no_grad():
        embedding_orig, tokens_orig = clip_model_vision(vision=images,
                                                        output_normalize=args.output_normalize, tokens=True)

    loss_inner_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, 'mean')
    args.norm = 'linf'
    args.iterations_adv = 100  # 10, 50
    args.stepsize_adv = 4 / 255  # 1 / 255
    images = pgd_veattack(
        forward=clip_model_vision,
        loss_fn=loss_inner_wrapper,
        data_clean=images,
        norm=args.norm,
        eps=args.eps,
        iterations=args.iterations_adv,
        stepsize=args.stepsize_adv,
        output_normalize=args.output_normalize,
        perturbation=torch.zeros_like(images).uniform_(-args.eps, args.eps).requires_grad_(True),
        mode='max',
        verbose=False
    )

    images = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )(images)

    # print(images.shape)

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
