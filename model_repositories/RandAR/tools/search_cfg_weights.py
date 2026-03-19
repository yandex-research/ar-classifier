""" Search for the optimal cfg weights for the given model.
    First using 10k samples to find the optimal value, then run on 50k samples to report.
"""
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_c2i_ddp.py
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
sys.path.append("./")
from RandAR.dataset.builder import build_dataset
from RandAR.utils.distributed import init_distributed_mode, is_main_process
from RandAR.dataset.augmentation import center_crop_arr
from RandAR.util import instantiate_from_config, load_safetensors
from RandAR.eval.fid import compute_fid


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sample_and_eval(tokenizer, gpt_model, cfg_scale, args, device, total_samples):
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % args.per_proc_batch_size == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    folder_name = (
        f"{args.exp_name}-{args.ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-"
        f"cfg-{cfg_scale:.2f}-seed-{args.global_seed}-num-{total_samples}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    iterations = int(samples_needed_this_gpu // args.per_proc_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    global_batch_size = args.per_proc_batch_size * dist.get_world_size()
    
    cur_iter = 0
    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (args.per_proc_batch_size,), device=device)
        cfg_scales = (1.0, cfg_scale)
    
        indices = gpt_model.generate(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        samples = tokenizer.decode_codes_to_img(indices, args.image_size_eval)
    
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        cur_iter += 1
        # I use this line to look at the initial images to check the correctness
        # comment this out if you want to generate more
        if args.debug:
            import pdb; pdb.set_trace()

        # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        sample_path = create_npz_from_sample_folder(sample_folder_dir, total_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

    fid, sfid, IS, precision, recall = compute_fid(args.ref_path, sample_path)
    return fid, sfid, IS, precision, recall


def main(args):
    # Setup PyTorch:
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    config = OmegaConf.load(args.config)
    # create and load model
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    tokenizer.load_state_dict(state_dict)

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    model_weight = load_safetensors(args.gpt_ckpt)
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()

    # Create folder to save samples:
    ckpt_string_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
    )
    args.ckpt_string_name = ckpt_string_name

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    dist.barrier()
    dist.destroy_process_group()
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples_search / global_batch_size) * global_batch_size)

    # CFG scales to be searched
    eval_results = {}
    cfg_scales_search = args.cfg_scales_search.split(",")
    cfg_scales_search = [float(cfg_scale) for cfg_scale in cfg_scales_search]
    cfg_scales_interval = float(args.cfg_scales_interval)
    cfg_scales_list = np.arange(cfg_scales_search[0], cfg_scales_search[1] + 1e-4, cfg_scales_interval)
    print(f"CFG scales to be searched: {cfg_scales_list}")

    result_file_name = (f"{args.results_path}/{args.exp_name}-{ckpt_string_name}-"
                        f"size-{args.image_size}-size-{args.image_size_eval}-search.json")

    # run throught the CFG scales
    for cfg_scale in cfg_scales_list:
        fid, sfid, IS, precision, recall = sample_and_eval(
            tokenizer, gpt_model, cfg_scale, args, device, total_samples)
        eval_results[f"{cfg_scale:.2f}"] = {
            "fid": fid,
            "sfid": sfid,
            "IS": IS,
            "precision": precision,
            "recall": recall
        }
        print(f"Eval results for CFG scale {cfg_scale:.2f}: {eval_results[f'{cfg_scale:.2f}']}")

        with open(result_file_name, "w") as f:
            json.dump(eval_results, f)
    
    # report the results
    total_samples = int(math.ceil(args.num_fid_samples_report / global_batch_size) * global_batch_size)
    optimal_cfg_scale = float(min(eval_results, key=lambda x: eval_results[x]["fid"]))
    fid, sfid, IS, precision, recall = sample_and_eval(
        tokenizer, gpt_model, optimal_cfg_scale, args, device, total_samples)
    
    print(f"Optimal CFG scale: {optimal_cfg_scale:.2f}")
    print(f"Eval results for optimal CFG scale: {fid, sfid, IS, precision, recall}")
    eval_results[f"{optimal_cfg_scale:.2f}-report"] = {
        "fid": fid,
        "sfid": sfid,
        "IS": IS,
        "precision": precision,
        "recall": recall
    }

    with open(result_file_name, "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sample results
    parser.add_argument("--config", type=str, default="configs/randar/randar_xl_0.7b.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input",)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scales-search", type=str, default="2.0, 8.0")
    parser.add_argument("--cfg-scales-interval", type=float, default=0.2)
    parser.add_argument("--sample-dir", type=str, default="/tmp")
    parser.add_argument("--num-inference-steps", type=int, default=88)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples-search", type=int, default=10000)
    parser.add_argument("--num-fid-samples-report", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--ref-path", type=str, default="/tmp/VIRTUAL_imagenet256_labeled.npz")
    # output results
    parser.add_argument("--results-path", type=str, default="./results")
    args = parser.parse_args()
    main(args)
