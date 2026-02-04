# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class VisualAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads,
                 bias=True, kdim=None, vdim=None):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, 'Only Support SelfAttention Currently'
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask = None):
        # query/key/value: [sq, b, h]
        sq, b, _ = query.size()

        assert query is key, 'Only Support Self-Attention Currently'
        sk = sq
        mixed_x_layer = self.in_proj(query)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(sq,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled, key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(b,
            self.num_attention_heads_per_partition,
            sq, self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)

        return output


class VisualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x




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

        self.model.load_state_dict(torch.load('/home/VEAttack/ckpt/fare_eps_4.pt', map_location='cpu'))
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



class VisionTransformer(nn.Module):

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            n_queries: int = 256,
            output_dim: int = 512,
            **kwargs
    ):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter((output_dim** -0.5) * torch.randn(output_dim, output_dim))

    def forward(self, x: torch.Tensor):
        x = x.to(
            dtype=self.conv1.weight.dtype,
            device=self.conv1.weight.device,
        )
        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + get_abs_pos(self.positional_embedding, x.size(1))

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj

        return x

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        clip_model_name = 'ViT-L-14'
        clip_model_orig, _, image_processor_clip = open_clip.create_model_and_transforms(
            clip_model_name, pretrained='openai'
        )
        images = images.cuda()
        # Remove the Normalize transform by creating a new Compose object
        normalize = image_processor_clip.transforms[-1]
        del image_processor_clip
        clip_model_orig.cpu()
        clip_model_vision = ClipVisionModel(model=clip_model_orig.visual, normalize=normalize)
        clip_model_vision.cuda()
        clip_model_orig.cuda()
        eps = 16 / 255
        output_normalize = False
        with torch.no_grad():

            embedding_orig, tokens_orig = clip_model_vision(vision=images,
                                                            output_normalize=output_normalize, tokens=True)

        loss_inner_wrapper = ComputeLossWrapper(embedding_orig, tokens_orig, 'mean')
        norm = 'linf'
        iterations_adv = 100  # 10, 50
        stepsize_adv = 4 / 255  # 1 / 255
        images = pgd_veattack(
            forward=clip_model_vision,
            loss_fn=loss_inner_wrapper,
            data_clean=images,
            norm=norm,
            eps=eps,
            iterations=iterations_adv,
            stepsize=stepsize_adv,
            output_normalize=output_normalize,
            perturbation=torch.zeros_like(images).uniform_(-eps, eps).requires_grad_(True),
            mode='max',
            verbose=False
        )
        images = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            )(images)
        del clip_model_orig

        return self(images)
