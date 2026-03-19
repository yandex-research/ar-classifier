# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
from torchvision import transforms
import os
import time
import argparse
import numpy as np
import wandb
from PIL import Image
from tqdm import tqdm
import shutil
import sys
sys.path.append("./")
import yaml
from omegaconf import OmegaConf, DictConfig
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from RandAR.util import instantiate_from_config, set_nested_key, load_safetensors, save_model_safetensors
from RandAR.dataset.builder import build_dataset
from RandAR.utils.visualization import make_grid
from RandAR.utils.logger import create_logger
from RandAR.model.generate import sample
from RandAR.utils.lr_scheduler import get_scheduler


def cycle(dl: torch.utils.data.DataLoader):
    # loop over the dataloader indefinitely
    while True:
        for data in dl:
            yield data


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    config = OmegaConf.load(args.config)

    #################### Accelerator ####################
    args.exp_name = args.exp_name + f'_bs_{config.global_batch_size}_lr_{config.optimizer.lr}'

    experiment_dir = os.path.join(args.results_dir, args.exp_name)
    accelerator_config = ProjectConfiguration(project_dir=experiment_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.accelerator.mixed_precision,
        log_with=config.accelerator.log_with,
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
    )
    set_seed(config.global_seed + accelerator.process_index)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(accelerator.state)
    else:
        logger = create_logger(None)

    #################### Data, Model, Optimization ####################
    dataset = build_dataset(is_train=True, args=args, transform=transforms.ToTensor())
    per_gpu_batch_size = int(
        config.global_batch_size
        // dist.get_world_size()
        // config.accelerator.gradient_accumulation_steps
    )
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    logger.info("Datasets contains {} samples for {} batches".format(len(dataset), len(dataset) // config.global_batch_size))

    model = instantiate_from_config(config.ar_model).to(accelerator.device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = instantiate_from_config(config.tokenizer).to(accelerator.device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    tokenizer.load_state_dict(state_dict)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    del ckpt

    optimizer = model.configure_optimizer(**config.optimizer)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=config.max_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )
    
    model.train()
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(model, optimizer, data_loader, lr_scheduler)
    data_loader = cycle(data_loader)

    total_iters = config.max_iters

    #################### Wandb Setup ####################
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="RandAR-Release",
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "config": dict(config),
                    "name": args.exp_name,
                    "dir": experiment_dir,
                }
            },
        )
    
    ################## Resume Training ##################
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("iters")]
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        ckpt_dir = f"{checkpoint_dir}/{saved_ckpt_dirs[-1]}/"
        if accelerator.is_main_process:
            logger.info(f"Resuming from {ckpt_dir}")
        accelerator.load_state(ckpt_dir)
        train_steps = int(saved_ckpt_dirs[-1].split("_")[-1])
    else:
        train_steps = 0

    #################### Training Loop ####################
    model.train()

    log_iters, running_loss, running_grad_norm, start_time = 0, 0, 0, time.time()

    logger.info(f"Starting training from iteration {train_steps} to {total_iters}")
    while train_steps < total_iters:
        model.train()
        x, y, inat_index = next(data_loader)
        x, y = x.to(accelerator.device, non_blocking=True), y.to(accelerator.device, non_blocking=True)
        image_tokens = x.reshape(x.shape[0], -1)
        cond = y.reshape(-1)

        with accelerator.accumulate(model):
            logits, loss, token_order = model(image_tokens, cond, targets=image_tokens)

            accelerator.backward(loss)
            if config.optimizer.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            
            grad_norm = 0.0
            for p in model.parameters():
                grad_norm += p.grad.data.norm(2).item()
            if grad_norm < config.optimizer.skip_grad_norm or train_steps < config.optimizer.skip_grad_iter:
                optimizer.step()
            
            optimizer.zero_grad()
            lr_scheduler.step()
            running_loss += (
                accelerator.gather(loss.repeat(per_gpu_batch_size)).mean().item() / config.accelerator.gradient_accumulation_steps
            )
            running_grad_norm += (
                grad_norm / config.accelerator.gradient_accumulation_steps
            )

        if accelerator.sync_gradients:
            log_iters += 1
            train_steps += 1
            model.eval()

            if train_steps % args.log_every == 0 and accelerator.is_main_process:
                # historical loss
                average_loss = torch.tensor(running_loss / args.log_every, device=accelerator.device).item()
                average_grad_norm = torch.tensor(running_grad_norm / args.log_every, device=accelerator.device).item()

                # speed
                end_time = time.time()
                average_time = (end_time - start_time) / args.log_every
                start_time = time.time()

                logger.info(f"Step {train_steps:08d} | Loss {average_loss:.4f} | Time {average_time:.4f}s | Grad Norm {average_grad_norm:.4f} | LR {lr_scheduler.get_last_lr()[0]:.5f}")
                running_loss = 0
                running_grad_norm = 0

                lr = optimizer.param_groups[0]['lr']

                logger_dict = {
                    "loss": average_loss,
                    "benchmark/time": average_time,
                    "grad_norm": average_grad_norm,
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                accelerator.log(logger_dict, step=train_steps)
            
            if train_steps % args.visualize_every == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    visualize_logits = logits[:args.visualize_num]
                    visualize_cond = cond[:args.visualize_num]
                    visualize_token_order = token_order[:args.visualize_num]
                    visualize_gt_indices = image_tokens[:args.visualize_num]
                    orig_token_order = torch.argsort(visualize_token_order)

                    img_token_num = logits.shape[1]
                    
                    # teacher forcing reconstruction
                    pred_recon_indices = torch.zeros(args.visualize_num, img_token_num, device=accelerator.device).long()
                    for i in range(img_token_num):
                        pred_recon_indices[:, i : i + 1] = torch.argmax(visualize_logits[:, i : i + 1], dim=-1)
                    pred_recon_indices = torch.gather(
                        pred_recon_indices.unsqueeze(-1),
                        dim=1,
                        index=orig_token_order.unsqueeze(-1)
                    ).squeeze(-1)
                    pred_recon_imgs = tokenizer.decode_codes_to_img(pred_recon_indices, args.image_size)

                    # vq reconstruction
                    gt_recon_indices = visualize_gt_indices
                    gt_recon_imgs = tokenizer.decode_codes_to_img(gt_recon_indices, args.image_size)

                    # generation
                    gen_indices = model.module.generate(
                        cond=visualize_cond,
                        token_order=None,
                        cfg_scales=[4.0, 4.0],
                        num_inference_steps=-1,
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                    )
                    model.module.remove_caches()
                    gen_imgs = tokenizer.decode_codes_to_img(gen_indices, args.image_size)

                    pred_recon_grid = make_grid(pred_recon_imgs)
                    gt_recon_grid = make_grid(gt_recon_imgs)
                    gen_grid = make_grid(gen_imgs)

                    accelerator.log({
                        "pred_recon": wandb.Image(pred_recon_grid),
                        "gt_recon": wandb.Image(gt_recon_grid),
                        "gen": wandb.Image(gen_grid),
                    }, step=train_steps)

            if train_steps % args.ckpt_every == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
                os.makedirs(ckpt_path, exist_ok=True)
                accelerator.save_state(ckpt_path)
                logger.info(f"Saved Iter {train_steps} checkpoint to {ckpt_path}")

                # remove redundantly more checkpoints
                for ckpt_dir in os.listdir(checkpoint_dir):
                    if ckpt_dir.startswith("iters") and ckpt_dir != f"iters_{train_steps:08d}":
                        save_iter = int(ckpt_dir.split("_")[-1])
                        if save_iter < train_steps - args.keep_last_k * args.ckpt_every:
                            if save_iter not in [5e4, 1e5, 2e5, 3e5]:
                                shutil.rmtree(os.path.join(checkpoint_dir, ckpt_dir))
                
                # copy the checkpoint to the disk location
                if args.disk_location:
                    disk_location = os.path.join(args.disk_location, args.exp_name)
                    # using try-catch to bypass random disk error or quota issues
                    try:
                        if os.path.exists(disk_location):
                            shutil.rmtree(disk_location)
                        shutil.copytree(checkpoint_dir, disk_location)
                        logger.info(f"Copied checkpoint to {disk_location}")
                    except Exception as e:
                        logger.error(f"Error copying checkpoint to {disk_location}: {e}")
            
            model.train()
            accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        accelerator.save_state(final_ckpt_dir)
        logger.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")
    
    accelerator.wait_for_everyone()
    logger.info("Training Done.")
    accelerator.end_training()

    # using shutil to copy the final checkpoint to the disk location
    if args.disk_location:
        disk_location = os.path.join(args.disk_location, args.exp_name)
        if os.path.exists(disk_location):
            shutil.rmtree(disk_location)
        shutil.copytree(checkpoint_dir, disk_location)
        logger.info(f"Copied final checkpoint to {disk_location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/randar/randar_xl_0.7b.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--ema", action="store_true", help="whether using ema training")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    # with 512 bs. 2.5k iters is 1 epoch
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)  # save every 5k iters
    # keep last k checkpoints; 1 means only keep the last checkpoint
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    # vq checkpoint
    parser.add_argument("--vq-ckpt", type=str, default="./checkpoints/vq_ds16_c2i.pt")
    # data
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--visualize-every", type=int, default=2000)
    parser.add_argument("--visualize-num", type=int, default=32)
    # wandb
    parser.add_argument("--wandb-entity", type=str, default="RandAR")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disk-location", type=str, default='')
    args = parser.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    main(args)