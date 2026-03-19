# RandAR GPT Model
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

import random
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .utils import DropPath, interleave_tokens, calculate_num_query_tokens_for_parallel_decoding
from .generate import sample
from .llamagen_gpt import LabelEmbedder, CaptionEmbedder, MLP, RMSNorm, \
    FeedForward, KVCache, find_multiple, apply_rotary_emb, precompute_freqs_cis_2d


def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(
        *x.shape[:-1], head_dim // 2, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        bs, xshaped.size(1), 1, xshaped.size(3), 2
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


""" Attention module modified for the parts updating KV cache
    Supporting slicing to accelerate inference
"""
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

        # this part is modified from LLaMAGen
        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        # this part is modified from LLaMAGen
        if self.kv_cache is not None:
            # [b, n_head, max_seq_len, head_dim]
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            
            # assuming that all the samples in a batch have the same input_pos
            max_pos = torch.max(input_pos) + 1
            keys = keys[:, :, :max_pos]
            values = values[:, :, :max_pos]
            if mask is not None:
                mask = mask[:, :, :, :max_pos]
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


""" Cloned from LLaMAGen: only the attention uses our customized version
"""
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


class RandARTransformer(nn.Module):
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
        position_order="random",
        num_inference_steps=88,
        zero_class_qk=True,
        grad_checkpointing=True,
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

        # RandAR related parameters
        self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.dim) * self.initializer_range)
        self.position_order = position_order
        self.num_inference_steps = num_inference_steps
        self.zero_class_qk = zero_class_qk
        self.grad_checkpointing = grad_checkpointing

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
    
    def remove_caches(self):
        for l in self.layers:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        token_order: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:
            return self.forward_train(idx, cond_idx, token_order, input_pos, targets, mask, valid)
        else:
            raise ValueError("idx and cond_idx cannot be both None")
        
    def forward_train(self,
                      idx: torch.Tensor,
                      cond_idx: torch.Tensor,
                      token_order: Optional[torch.Tensor] = None,
                      input_pos: Optional[torch.Tensor] = None,
                      targets: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None,
                      valid: Optional[torch.Tensor] = None,):
        """ Args:
            idx: [bsz, seq_len] GT image tokens for teacher forcing
            cond_idx: [bsz, cls_token_num] Cls tokens
            token_order: [bsz, seq_len] Position order for each token
            input_pos: [seq_len] Position index for each token (default None)
            targets: [bsz, seq_len] Target tokens for teacher forcing (default None)
            mask: [bsz, seq_len, seq_len] Causal mask for attention (default None)
            valid: [bsz, seq_len] Valid mask for loss calculation (default None)
        """
        # 1. Prepare orders
        bs = idx.shape[0]
        if token_order is None:
            if self.position_order == "random":
                token_order = torch.arange(self.block_size, device=self.tok_embeddings.weight.device, dtype=torch.long)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
                token_order = token_order.contiguous()
            elif self.position_order == "raster":
                token_order = torch.arange(self.block_size, device=idx.device)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                token_order = token_order.contiguous()
            else:
                raise ValueError(f"Invalid position order: {self.position_order}")
        
        # permute the image tokens according to the random order
        idx = torch.gather(idx.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous() # [bsz, seq_len]
        targets = torch.gather(targets.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous() # [bsz, seq_len]

        # 2. Prepare embeddings and freqs_cis
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
            :, : self.cls_token_num
        ] # [bsz, cls_token_num, dim]

        token_embeddings = self.tok_embeddings(idx)
        token_embeddings = self.tok_dropout(token_embeddings) # [bsz, seq_len, dim]
        position_instruction_tokens = self.get_position_instruction_tokens(token_order) # [bsz, seq_len, dim]

        h = torch.cat(
            (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
            dim=1
        )
        
        token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        freqs_cis = torch.cat(
            (self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)),
            dim=1
        )

        # 3. Forward
        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)
        
        h = self.norm(h)
        logits = self.output(h).float()
        token_logits = logits[:, self.cls_token_num::2].contiguous()

        # 4. Loss computation
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), targets.view(-1))

        return token_logits, loss, token_order
    
    def forward_inference(self, 
                          x: torch.Tensor, 
                          freqs_cis: torch.Tensor, 
                          input_pos: torch.Tensor):
        """ Args:
            x: [bs, query_num, dim] Input tokens
            freqs_cis: [bs, query_num, n_head, dim // n_head] Frequency embeddings
            input_pos: [query_num] Position index for each token
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    def get_position_instruction_tokens(self, token_order):
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.n_head, self.dim // self.n_head)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], self.block_size, 1, 1) # [1, block_size, n_head, dim // n_head]
        
        # apply rotary embedding
        position_instruct_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], self.block_size, self.dim).contiguous()
        return position_instruct_tokens

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

    # large changes to LLaMAGen, directly supporting parallel decoding controlled by num_inference_steps
    def generate(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """ Args:
            cond: [bsz, seq_len] Conditional tokens
            token_order: [bsz, seq_len] Position order for each token
            cfg_scales: Tuple (cfg_scale_start, cfg_scale_end) linear cfg scales, set start=end for constant cfg_scale
            num_inference_steps: int Number of inference steps, set to -1 or the number of image tokens to disable parallel decoding
            temperature: float Temperature for sampling
            top_k: int Top-k for sampling
            top_p: float Top-p for sampling
        """
        bs = cond.shape[0]
        
        # Step-1: Generate the token orders and result sequences
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)
        
        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)
        
        # Step-2: Prepare the freqs_cis and position_instruction_tokens
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # Step-3: Prepare CFG
        if cfg_scales[-1] > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)
    
        # Step-4: KV Cache setup
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)

        # Step-5: Autoregressive generation with parallel decoding
        if num_inference_steps == -1:
            # if -1, one token at a time, no parallel decoding
            num_inference_steps = self.block_size
        
        cur_inference_step = 0
        num_query_token_cur_step = 1 # how many tokens to decode at this step
        query_token_idx_cur_step = 0 # the index of the first token to decode at this step

        # Step 5-1: Prepare the first step
        # [cls_token, query_token_0, ..., query_token_n]
        x = torch.cat([cond_combined_tokens, 
                       position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                       dim=1)
        cur_freqs_cis = torch.cat([self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), 
                                   img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                                   dim=1)
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        # Step 5-2: Start the loop
        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:
            # Step 5-3: Decode the current step tokens
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)

            # apply CFG
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # query tokens' logits and indices
            logits = logits[:, -num_query_token_cur_step:] # [bs, query_num, vocab_size]
            indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long, device=cond.device)
            for i in range(num_query_token_cur_step):
                indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]
            
            # save the result tokens
            result_indices[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()
            
            img_tokens = self.tok_embeddings(indices)
            if cfg_scales[-1] > 1.0:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # Step 5-4: Prepare for the next step
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size, 
                query_token_idx_cur_step, num_query_token_cur_step)
            
            ########## Important: Prepare the tokens ##########
            # [cur_img_0, cur_query_1, ..., cur_query_n, cur_img_n, next_query_0, ..., next_query_m]
            x = torch.zeros(bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, self.dim, dtype=x.dtype, device=cond.device)
            
            # cur_img_0
            x[:, :1] = img_tokens[:, :1] 
            
            # [cur_query_1, ..., cur_query_n]
            cur_query_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens
            
            # [cur_img_1, ..., cur_img_n]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1 : num_query_token_cur_step]
            
            # [next_query_0, ..., next_query_m]
            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step
            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            x[:, 2 * num_query_token_cur_step - 1 :] = next_position_instruction_tokens

            ########## Important: Prepare the freqs_cis ##########
            cur_freqs_cis = torch.zeros((bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *self.freqs_cis.shape[-2:]), 
                                         dtype=cur_freqs_cis.dtype, device=cond.device)
            
            # cur_img_0
            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + 1]

            # [cur_query_1, ..., cur_query_n]
            cur_query_freq_cis = img_token_freq_cis[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis

            # [cur_img_1, ..., cur_img_n]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

            # [next_query_0, ..., next_query_m]
            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 2 * num_query_token_cur_step - 1 :] = next_freq_cis

            # Step 5-5: Move the query pointer idx
            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break
            
            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step] # position of cur_query_0
            input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
            num_query_token_cur_step = num_query_token_next_step
        
        # Step 6: Return to raster order for tokenizer decoding
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
        return result_indices
