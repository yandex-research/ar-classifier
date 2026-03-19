import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/RandAR")
sys.path.append(".")
from RandAR.util import instantiate_from_config

from image_preprocessing import (
    calc_statistics,
    center_crop_arr,
    random_crop_arr,
    sample_imagenet,
)


@torch.no_grad()
def calculate_likelihoods(
    rank,
    tokenizer,
    model,
    data_path,
    classes,
    batch_size,
    n_trials,
    g,
    use_raster,
    t=0,
    valid=False,
):

    if "maskgit" in config.tokenizer.target:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        )

    device = model.cls_embedding.embedding_table.weight.device
    n_class = len(classes)
    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)
    sum_logprobs = torch.zeros((len_dataset, n_class, args.image_size)).to(device)

    targets_ = torch.zeros(len_dataset).long().to(device)
    img_num = 0

    cnt = 0
    true = 0

    for class_ in classes:
        i = 0
        file_name = f"{data_path}/{class_}_{i}.JPEG"

        while os.path.exists(file_name):

            orig_img = PIL.Image.open(file_name).convert("RGB")

            likelyhoods_local = torch.zeros(n_class).to(device)

            for trial in range(n_trials):
                if args.use_augmentations:
                    img = random_crop_arr(orig_img, args.image_size, g=g)
                else:
                    img = center_crop_arr(orig_img, args.image_size)

                img = transform(img).to(device).unsqueeze(0)

                if "maskgit" in config.tokenizer.target:
                    codes = tokenizer.encode_indices(img)
                    latents = codes.reshape(img.shape[0], -1)[0]
                    latents = latents.tile((batch_size, 1))
                else:
                    h = tokenizer.encoder(img)
                    h = tokenizer.quant_conv(h)
                    h = (1 - t) * h + torch.randn_like(h) * t
                    quant, emb_loss, info = tokenizer.quantize(h)
                    latents = info[2]
                    latents = latents.tile((batch_size, 1))

                token_order = torch.randperm(gpt_model.block_size, generator=g).to(
                    device
                )
                if use_raster:
                    token_order = torch.arange(gpt_model.block_size).to(device)
                if args.use_const_random:
                    g_ = torch.Generator()
                    g_.manual_seed(args.seed)
                    token_order = torch.randperm(gpt_model.block_size, generator=g_).to(
                        device
                    )

                token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
                token_order = token_order.contiguous()

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]
                    with torch.autocast("cuda", torch.float):
                        logits, loss, token_order = gpt_model(latents, cond, targets=latents, token_order=token_order)
                        targets = (torch.gather(latents.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous())
                    prob_seq = F.softmax(logits, dim=-1)

                    log_prob = (prob_seq[torch.arange(latents.shape[0])[:, None], torch.arange(latents.shape[1])[None, :], targets]).log()
                    likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = log_prob.sum(dim=-1)
                    sum_logprobs[img_num, batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob
                    likelyhoods_local[batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob.sum(dim=-1)

            dist.all_reduce(likelyhoods_local, op=dist.ReduceOp.SUM)

            dist.barrier()

            cnt += 1
            if likelyhoods_local.argmax(dim=-1) == class_:
                true += 1

            if rank == 0 and not valid:
                print(class_, true / cnt)

            i += 1
            file_name = f"{data_path}/{class_}_{i}.JPEG"

            targets_[img_num] = class_
            img_num += 1

    dist.all_reduce(likelyhoods, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_logprobs, op=dist.ReduceOp.SUM)
    return likelyhoods, targets_, sum_logprobs


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--config", type=str, default="model_repositories/RandAR/configs/randar/randar_l_0.3b_llamagen.yaml")
parser.add_argument("--downsample_size", type=int, default=16)


parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)
parser.add_argument("--gpt_ckpt", type=str, default="model_iters_00360000.pt")
parser.add_argument("--vq_ckpt", type=str, default="model_weights/RandAR_weights/vq_ds16_c2i.pt")
parser.add_argument("--n_samples", type=int, default=2)
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_raster", type=bool, default=False)
parser.add_argument("--use_const_random", type=bool, default=False)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--t", type=float, default=0.0)


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
config = OmegaConf.load(args.config)
tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
ckpt = torch.load(args.vq_ckpt, map_location="cpu")
if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt
tokenizer.load_state_dict(state_dict)
tokenizer.eval()

latent_size = args.image_size // args.downsample_size
gpt_model = instantiate_from_config(config.ar_model).to(
    device=device, dtype=torch.bfloat16
)

model_weight = torch.load(args.gpt_ckpt)
gpt_model.load_state_dict(model_weight, strict=True)
gpt_model.eval()
dist.barrier()


############################## CALCULATE LIKELIHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, targets, sum_logprobs = calculate_likelihoods(
    rank,
    tokenizer,
    gpt_model,
    data_path,
    classes,
    args.batch_size,
    args.n_trials,
    g,
    args.use_raster,
    args.t,
)

############################## LOG METRICS ##############################

if rank == 0:
    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_0_{i}_trials:", acc_i_0.mean(), "+-", acc_i_0.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_1_{i}_trials:", acc_i_1.mean(), "+-", acc_i_1.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_2_{i}_trials:", acc_i_2.mean(), "+-", acc_i_2.std())

    print("========================================")

dist.destroy_process_group()
