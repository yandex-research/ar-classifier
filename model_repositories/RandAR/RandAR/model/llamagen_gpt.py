# Base Model Cloned from LLaMAGEN
# Not used in our RandAR
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import DropPath
from .generate import (
    top_k_top_p_filtering,
    sample,
    logits_to_probs,
    prefill,
    decode_one_token,
    decode_n_tokens,
)


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = "c2i"

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
        )
        self.register_buffer(
            "uncond_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, ffn_dim_multiplier: int, multiple_of: int, ffn_dropout_p: float
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        during inference:
        Args:
            x: [bsz, seqlen, dim], input tensor.
            freqs_cis: [bsz, seqlen, head_dim // 2, 2], used to apply rotary emb.
            input_pos: [seqlen], used to update KV cache.
            mask: [bsz, seqlen, seqlen], used to mask out attention weights.
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=(
                True if mask is None else False
            ),  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        initializer_range=0.02,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
        num_classes=1000,
        caption_dim=2048,
        class_dropout_prob=0.1,
        model_type="c2i",
        vocab_size=16384,
        cls_token_num=1,
        block_size=256,
        max_batch_size=32,
        max_seq_len=2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.cls_token_num = cls_token_num
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(caption_dim, dim, class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                    token_dropout_p=token_dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    drop_path=dpr[layer_id],
                )
            )

        # output layer
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # initialization
        self.initializer_range = initializer_range
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_head, head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:  # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                :, : self.cls_token_num
            ]
            idx = idx[:, :-1]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:  # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                    :, : self.cls_token_num
                ]
            else:  # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        if self.training:
            freqs_cis = self.freqs_cis[: token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        if self.training:
            logits = logits[:, self.cls_token_num - 1 :].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # append one output to align the num of outs with other models
        return logits, loss, None

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def configure_optimizer(
        self, lr, weight_decay, beta1, beta2, max_grad_norm, **kwargs
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        import inspect

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **extra_args
        )
        return optimizer

    def generate(
        self,
        cond,
        max_new_tokens,
        emb_masks=None,
        cfg_scale=1.0,
        cfg_interval=-1,
        **sampling_kwargs
    ):
        if self.model_type == "c2i":
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond) * self.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        elif self.model_type == "t2i":
            if cfg_scale > 1.0:
                cond_null = torch.zeros_like(cond) + self.cls_embedding.uncond_embedding
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = cond.shape[1]
        else:
            raise Exception("please check model type")

        T_new = T + max_new_tokens
        max_seq_length = T_new
        max_batch_size = cond.shape[0]

        device = cond.device
        with torch.device(device):
            max_batch_size_cfg = (
                max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
            )
            self.setup_caches(
                max_batch_size=max_batch_size_cfg,
                max_seq_length=max_seq_length,
                dtype=self.tok_embeddings.weight.dtype,
            )

        if emb_masks is not None:
            assert emb_masks.shape[0] == max_batch_size
            assert emb_masks.shape[-1] == T
            if cfg_scale > 1.0:
                self.causal_mask[:, :, :T] = self.causal_mask[:, :, :T] * torch.cat(
                    [emb_masks, emb_masks]
                ).unsqueeze(1)
            else:
                self.causal_mask[:, :, :T] = self.causal_mask[
                    :, :, :T
                ] * emb_masks.unsqueeze(1)

            eye_matrix = torch.eye(
                self.causal_mask.size(1), self.causal_mask.size(2), device=device
            )
            self.causal_mask[:] = self.causal_mask * (1 - eye_matrix) + eye_matrix

        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

        input_pos = torch.arange(0, T, device=device)
        next_token = prefill(
            self, cond_combined, input_pos, cfg_scale, **sampling_kwargs
        )
        seq[:, T : T + 1] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        generated_tokens, _ = decode_n_tokens(
            self,
            next_token,
            input_pos,
            max_new_tokens - 1,
            cfg_scale,
            cfg_interval,
            **sampling_kwargs
        )
        seq[:, T + 1 :] = torch.cat(generated_tokens, dim=1)

        return seq[:, T:]


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
# def GPT_7B(**kwargs):
#     return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

# def GPT_3B(**kwargs):
#     return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

# def GPT_1B(**kwargs):
#     return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

# ### class-conditional
# def GPT_XXXL(**kwargs):
#     return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

# def GPT_XXL(**kwargs):
#     return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

# def GPT_XL(**kwargs):
#     return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

# def GPT_L(**kwargs):
#     return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

# def GPT_B(**kwargs):
#     return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M


# GPT_models = {
#     'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
#     'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
# }
