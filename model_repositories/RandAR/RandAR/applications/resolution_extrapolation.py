from dataclasses import dataclass
from typing import Optional, List, Tuple
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from RandAR.model.randar_gpt import RandARTransformer, batch_apply_rotary_emb
from RandAR.model.llamagen_gpt import apply_rotary_emb
from RandAR.model.generate import sample
from RandAR.model.utils import calculate_num_query_tokens_for_parallel_decoding


def precompute_freqs_cis_2d_extrapolation(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, extrapolation_factor: float = 2.0,
    ntk_boundary: int = 2
):
    """ Generating the rotary \PE for extrapolation
        ntk_boundary: where to separate the interpolated and extrapolated PE
    """
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    
    power = torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim
    freqs = 1.0 / (
        base ** power
    )
    freqs_extra = freqs.clone() # extrapolated freqs
    freqs_inter = freqs.clone() # interpolated freqs

    base_max_pos = int(grid_size)
    t = torch.linspace(0, base_max_pos - 1, grid_size, device=freqs.device)
    freqs_extra = torch.outer(t, freqs_extra)  # (real_grid_size, head_dim // 2)

    base_max_pos = int(grid_size // extrapolation_factor)
    t = torch.linspace(0, base_max_pos - 1, grid_size, device=freqs.device)
    freqs_inter = torch.outer(t, freqs)  # (real_grid_size, head_dim // 2)

    # IMPORTANT: mix the frequencies inspired by NTK
    freqs = freqs_extra.clone()
    # Change low-frequency part to interpolated PE
    freqs[:, ntk_boundary:] = freqs_inter[:, ntk_boundary:]

    # Generate the rotary embeddings
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
    cond_freqs = torch.zeros(cls_token_num, n_elem // 2, 2)
    cond_cache = torch.cat((cond_freqs, cache), dim=0)
    return cond_cache


def token_order_generator(
    bs, grid_size, mode="random", extrapolation_factor=2.0
):
    """
    The order for AR in resolution extrapolation
    mode: choose from "random" or "hierarchical"
    By default, we deal with an extrapolation factor of 2.0. This function might need changes for other extrapolation factors.
    """
    if mode == "random":
        token_order = torch.arange(grid_size ** 2, device=cond.device)
        token_order = token_order.unsqueeze(0).repeat(bs, 1)
        token_order = token_order.contiguous()
        for i in range(bs):
            token_order[i] = token_order[i][torch.randperm(grid_size ** 2)]
        token_order = token_order.contiguous()
    elif mode == "hierarchical":
        grid = np.arange(grid_size ** 2).reshape(grid_size, grid_size)
        even_indices = grid[::2, ::2].flatten()
        other_indices = np.array([i for i in grid.flatten() if i not in even_indices])
        even_indices = torch.tensor(even_indices)
        other_indices = torch.tensor(other_indices)
        even_indices = even_indices.repeat(bs, 1)
        other_indices = other_indices.repeat(bs, 1)

        for i in range(bs):
            even_indices[i] = even_indices[i][torch.randperm(len(even_indices[i]))]
            other_indices[i] = other_indices[i][torch.randperm(len(other_indices[i]))]
        token_order = torch.cat([even_indices, other_indices], dim=1).contiguous()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return token_order


def get_position_instruction_tokens(
    token_order: torch.Tensor,
    position_instruct_freqs_cis: torch.Tensor,
    model: RandARTransformer,
):
    position_instruct_tokens = model.pos_instruct_embeddings.view(1, 1, model.n_head, model.dim // model.n_head)
    position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], token_order.shape[1], 1, 1)
    position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
    position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], token_order.shape[1], model.dim).contiguous()
    return position_instruct_tokens


def generate_resolution_extrapolation(
    model: RandARTransformer,
    cond: torch.Tensor,
    token_order: torch.Tensor = None,
    cfg_scales: Tuple[float, float] = (1.0, 1.0),
    spatial_cfg_scales: Tuple[float, float] = (1.0, 1.0),
    spatial_masking_ratio: float = 0.5,
    num_inference_steps: int = 512,
    extrapolation_factor: float = 2.0,
    ntk_boundary: int = 2,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    debug: bool = False,
):
    bs = cond.shape[0]
    grid_size = int((model.block_size ** 0.5) * extrapolation_factor)
    block_size = grid_size ** 2

    # Step-1: Generate the token orders and result sequences (Same as regular RandAR)
    if token_order is None:
        token_order = token_order_generator(bs, grid_size, mode="hierarchical", extrapolation_factor=2.0)
    
    result_indices = torch.zeros((bs, block_size), dtype=torch.long, device=cond.device)

    # Step-2: Fill in the low-resolution tokens
    low_res_token_num = int(block_size / (extrapolation_factor ** 2))
    low_res_token_order = token_order[:, :low_res_token_num].clone()
    low_res_grid_size, low_res_block_size = int(grid_size / extrapolation_factor), int(grid_size / extrapolation_factor) ** 2
    low_res_token_y = ((low_res_token_order // grid_size) / extrapolation_factor).to(torch.long)
    low_res_token_x = ((low_res_token_order % grid_size) / extrapolation_factor).to(torch.long)
    low_res_token_x = low_res_token_x.contiguous()
    low_res_token_y = low_res_token_y.contiguous()
    low_res_token_order = low_res_token_x + low_res_token_y * low_res_grid_size
    low_res_token_order = low_res_token_order.to(cond.device)
    low_res_indices = model.generate(
        cond=cond,
        token_order=low_res_token_order,
        cfg_scales=cfg_scales,
        num_inference_steps=num_inference_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    orig_low_res_indices = torch.gather(low_res_indices.unsqueeze(-1), 1, low_res_token_order.unsqueeze(-1)).squeeze(-1)
    result_indices[:, :low_res_token_num] = orig_low_res_indices

    # Step-3: Prepare the freqs_cis and position_instruction_tokens (Same as regular RandAR)
    freqs_cis = precompute_freqs_cis_2d_extrapolation(
        grid_size = grid_size, n_elem = model.dim // model.n_head, 
        base=model.rope_base, cls_token_num=model.cls_token_num, 
        extrapolation_factor=extrapolation_factor, ntk_boundary=ntk_boundary)
    freqs_cis = freqs_cis.to(cond.device)
    position_instruct_freqs_cis = freqs_cis[model.cls_token_num:].clone()
    img_token_freq_cis = position_instruct_freqs_cis[token_order].to(cond.device).contiguous()

    position_instruction_tokens = get_position_instruction_tokens(
        token_order=token_order, position_instruct_freqs_cis=img_token_freq_cis, model=model)
    
    # Step-4: Prepare for CFG
    if cfg_scales[-1] > 1.0 and spatial_cfg_scales[-1] > 1.0:
        # both cls cfg and spatial cfg are applied
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null, cond_null])
        img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis, img_token_freq_cis])
        position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens, position_instruction_tokens])

        cfg_bs = 3 * bs
    
    elif cfg_scales[-1] > 1.0:
        # only cls cfg is applied
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
        img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
        position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
        cfg_bs = 2 * bs
    else:
        cond_combined = cond
        cfg_bs = bs
    cond_combined_tokens = model.cls_embedding(cond_combined, train=False)

    # Step-5: KV Cache setup
    max_seq_len = cond_combined_tokens.shape[1] + block_size * 2
    with torch.device(cond.device):
        model.setup_caches(max_batch_size=cfg_bs, max_seq_length=max_seq_len, dtype=model.tok_embeddings.weight.dtype)

    # Step-6: Autoregressive generation with parallel decoding
    if num_inference_steps == -1:
        num_inference_steps = block_size
    
    cur_inference_step = 0
    num_query_token_cur_step = 1
    query_token_idx_cur_step = low_res_token_num # start from the high-resolution tokens

    # Step 6-1: Prepare the first step
    # We will first put the low resolution tokens as context.
    # [cls_token, low_res_query_token_0, low_res_img_token_0, ..., high_res_query_token_0, ..., high_res_query_token_n, high_res_img_token_n]
    low_res_position_instruction_tokens = position_instruction_tokens[:, :low_res_token_num]
    low_res_img_tokens = model.tok_embeddings(orig_low_res_indices)

    if cfg_scales[-1] > 1.0 and spatial_cfg_scales[-1] > 1.0:
        # randomly masking for spatial cfg
        low_res_img_tokens_dropout = torch.zeros_like(low_res_img_tokens)
        mask = torch.ones(low_res_img_tokens.shape[0], low_res_img_tokens.shape[1], device=low_res_img_tokens.device, dtype=low_res_img_tokens.dtype)
        for i in range(low_res_img_tokens.shape[0]):
            idx = torch.randperm(low_res_img_tokens.shape[1])[:int(low_res_img_tokens.shape[1] * spatial_masking_ratio)]
            mask[i, idx] = 0
        low_res_img_tokens_dropout = low_res_img_tokens * mask.unsqueeze(-1)
        low_res_img_tokens = torch.cat([low_res_img_tokens, low_res_img_tokens, low_res_img_tokens_dropout], dim=0)
        
    elif cfg_scales[-1] > 1.0:
        low_res_img_tokens = torch.cat([low_res_img_tokens, low_res_img_tokens], dim=0)
    else:
        pass
    
    low_res_pi_img_tokens = torch.zeros_like(torch.cat([low_res_img_tokens, low_res_img_tokens], dim=1))
    low_res_pi_img_tokens[:, ::2] = low_res_position_instruction_tokens
    low_res_pi_img_tokens[:, 1::2] = low_res_img_tokens

    low_res_rope_freqs_cis = torch.zeros_like(torch.cat([
        img_token_freq_cis[:, :low_res_token_num],
        img_token_freq_cis[:, :low_res_token_num],
    ], dim=1))
    low_res_rope_freqs_cis[:, ::2] = img_token_freq_cis[:, :low_res_token_num]
    low_res_rope_freqs_cis[:, 1::2] = img_token_freq_cis[:, :low_res_token_num]

    x = torch.cat([cond_combined_tokens, 
                   low_res_pi_img_tokens,
                   position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                   dim=1)
    cur_freqs_cis = torch.cat([freqs_cis[:model.cls_token_num].unsqueeze(0).repeat(cfg_bs, 1, 1, 1), 
                               low_res_rope_freqs_cis,
                               img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], 
                               dim=1)
    input_pos = torch.arange(0, x.shape[1], device=cond.device)

    if debug:
        pbar = tqdm(total=block_size)
        for i in range(query_token_idx_cur_step):
            pbar.update(1)

    # Step 6-2: Start the loop
    while query_token_idx_cur_step < block_size - num_query_token_cur_step and query_token_idx_cur_step < block_size - 1:
        # Step 6-3: Decode the current step tokens
        logits = model.forward_inference(x, cur_freqs_cis, input_pos)

        # apply CFG
        if cfg_scales[-1] > 1.0 and spatial_cfg_scales[-1] > 1.0:
            cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / block_size
            cur_spatial_cfg_scale = spatial_cfg_scales[0] + (spatial_cfg_scales[-1] - spatial_cfg_scales[0]) * query_token_idx_cur_step / block_size
            
            cond_logits, spatial_cond_logits, uncond_logits = torch.chunk(logits, 3, dim=0)
            logits = uncond_logits + cur_cfg_scale * (cond_logits - spatial_cond_logits) + cur_spatial_cfg_scale * (spatial_cond_logits - uncond_logits)
        elif cfg_scales[-1] > 1.0:
            cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / block_size
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)
        else:
            pass
        
        # query tokens' logits and indices
        logits = logits[:, -num_query_token_cur_step:] # [bs, query_num, vocab_size]
        indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long, device=cond.device)
        for i in range(num_query_token_cur_step):
            indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]
        
        # save the result tokens
        result_indices[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()
        if debug:
            pbar.update(num_query_token_cur_step)
        img_tokens = model.tok_embeddings(indices)
        if cfg_scales[-1] > 1.0 and spatial_cfg_scales[-1] > 1.0:
            img_tokens = torch.cat([img_tokens, img_tokens, img_tokens], dim=0)
        elif cfg_scales[-1] > 1.0:
            img_tokens = torch.cat([img_tokens, img_tokens], dim=0)
        else:
            pass
        
        # Step 6-4: Prepare for the next step
        cur_inference_step += 1
        num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
            cur_inference_step, num_inference_steps, block_size, 
            query_token_idx_cur_step, num_query_token_cur_step)

        ########## Important: Prepare the tokens ##########
        # [cur_img_0, cur_query_1, ..., cur_query_n, cur_img_n, next_query_0, ..., next_query_m]
        x = torch.zeros(cfg_bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, model.dim, dtype=x.dtype, device=cond.device)
        
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
        cur_freqs_cis = torch.zeros((cfg_bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *position_instruct_freqs_cis.shape[-2:]), 
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
        if query_token_idx_cur_step > block_size:
            break
        
        last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step] # position of cur_query_0
        input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
        num_query_token_cur_step = num_query_token_next_step
    
    if debug:
        pbar.close()
        
    # Step 7: Return to raster order for tokenizer decoding
    reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1).to(result_indices.device)
    result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
    return result_indices

