import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/SiT")
sys.path.append(".")
from models import SiT_models

from image_preprocessing import (
    calc_statistics,
    center_crop_arr,
    random_crop_arr,
    sample_imagenet,
)


@torch.no_grad()
def calculate_likelihoods(
    rank, vae, model, diffusion, data_path, classes, batch_size, step_size, g
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    device = model.y_embedder.embedding_table.weight.device

    n_class = len(classes)
    n_trials = 1000 // step_size

    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    ts = np.arange(step_size // 2, 1000, step_size) / 1000

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)
    targets_ = torch.zeros(len_dataset).long().to(device)
    img_num = 0

    cnt = 0
    true = 0

    for class_ in classes:

        i = 0
        file_name = f"{data_path}/{class_}_{i}.JPEG"

        while os.path.exists(file_name):

            local_likelyhoods = torch.zeros(n_class).to(device)

            for trial, t in enumerate(ts):

                orig_img = PIL.Image.open(file_name).convert("RGB")

                if args.use_augmentations:
                    img = random_crop_arr(orig_img, args.image_size, g=g)
                else:
                    img = center_crop_arr(orig_img, args.image_size)

                img = transform(img).to(device).unsqueeze(0)
                latents = vae.encode(img).latent_dist.sample().mul_(0.18215)

                batch_ts = torch.tensor([t]).long().to(device)
                noise = torch.randn(latents.shape, generator=g).to(device)
                noised_latents = t * latents + (1 - t) * noise
                noised_latents = noised_latents.tile((batch_size, 1, 1, 1))

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                    with torch.autocast(device.type, torch.bfloat16):
                        u = model(noised_latents, batch_ts, y=cond)
                        noise_pred = noised_latents - u * t

                    loss = ((noise - noise_pred) ** 2).sum(dim=(1, 2, 3))

                    likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = -loss
                    local_likelyhoods[batch_ind * batch_size : (batch_ind + 1) * batch_size] += -loss

            dist.all_reduce(local_likelyhoods, op=dist.ReduceOp.SUM)

            cnt += 1
            if YS[local_likelyhoods.argmax()] == class_:
                true += 1

            i += 1
            file_name = f"{data_path}/{class_}_{i}.JPEG"

            targets_[img_num] = class_
            img_num += 1

            if rank == 0:
                print(img_num, true / cnt)

    dist.all_reduce(likelyhoods, op=dist.ReduceOp.SUM)
    return likelyhoods, targets_


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--model", type=str, default="SiT-XL/2")
parser.add_argument("--downsample_size", type=int, default=8)

parser.add_argument("--sit_ckpt", type=str, default="model_weights/DiT_weights/DiT-XL-2-256x256.pt")

parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)

parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--step_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_augmentations", type=bool, default=False)

args = parser.parse_args()

g = torch.Generator()
g.manual_seed(args.seed)


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = rank
torch.cuda.set_device(device)
print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

dist.barrier()

############################## CREATE DATA ##############################

data_path = f"imagenet_data/imagenet_{args.dataset}"
if rank == 0:
    folder = Path(data_path)
    if folder.exists():
        shutil.rmtree(folder)

    classes = sample_imagenet(
        args.imagenet_X_path,
        args.imagenet_val_path,
        data_path,
        N_SAMPLES=args.n_samples,
    )
    np.save(f"{data_path}", classes)
dist.barrier()

classes = torch.tensor(np.load(f"{data_path}.npy")).to(device)

############################## INIT MODELS ##############################

tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
tokenizer.eval()
dist.barrier()

model = SiT_models[args.model](
    input_size=args.image_size // args.downsample_size, num_classes=1000
).to(device)

ckpt = torch.load(args.sit_ckpt, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()
model.to(torch.bfloat16)
dist.barrier()

############################## CALCULATE LIKELYHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, targets = calculate_likelihoods(
    rank, tokenizer, model, None, data_path, classes, args.batch_size, args.step_size, g
)

############################## LOG METRICS ##############################

n_trials = 1000 // args.step_size

if rank == 0:
    for i in range(1, n_trials + 1):
        acc_i = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_{i}_trials:", acc_i)

dist.destroy_process_group()
