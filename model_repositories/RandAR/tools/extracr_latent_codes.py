# Modified from LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_c2i.py
# Modified from fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py

from omegaconf import OmegaConf
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append("./")

from RandAR.dataset.builder import build_dataset
from RandAR.utils.distributed import init_distributed_mode, is_main_process
from RandAR.dataset.augmentation import center_crop_arr
from RandAR.util import instantiate_from_config


def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."
    # Setup DDP:
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # create and load tokenizer model
    config = OmegaConf.load(args.config)
    vq_model = instantiate_from_config(config.model).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    vq_model.load_state_dict(state_dict)

    # setup data augmentation
    crop_size = args.image_size 
    if args.aug_mode == 'ten-crop': # default choice of llamagen
        crop_size = int(args.image_size * args.crop_range)
        if args.tokenizer_name == 'maskgit': # [0, 1]
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    elif args.aug_mode == 'adm': # default choice of ADM and MAR
        if args.tokenizer_name == 'maskgit':
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    is_train = (not args.debug) # using val.tar for debugging
    dataset = build_dataset(is_train=is_train, args=args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # iterating
    if is_main_process():
        pbar = tqdm(total=len(data_loader))
        for i in range(dataset.nb_classes):
            os.makedirs(f'{args.code_path}/{args.dataset}-{args.tokenizer_name}-{args.aug_mode}-{args.image_size}_codes/{i}', exist_ok=True)

    for x, y, index in data_loader:
        x = x.to(device)
        if args.aug_mode == 'ten-crop':
            x_all = x.flatten(0, 1)
            num_aug = 10
        elif args.aug_mode == 'adm' :
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            x_all[::2] = x
            x_all[1::2] = x_flip
            num_aug = 2
        else:
            x_all = x
            num_aug = 1
        
        y = y.to(device)
        with torch.no_grad():
            indices = vq_model.encode_indices(x_all)
        codes = indices.reshape(x.shape[0], num_aug, -1)

        # using index to indicate file name
        index = index.cpu().numpy()

        x = codes.detach().cpu().numpy()    # (bs, num_aug, args.image_size//16 * args.image_size//16)
        y = y.detach().cpu().numpy()        # (bs,)

        for i in range(x.shape[0]):
            np.save(f'{args.code_path}/{args.dataset}-{args.tokenizer_name}-{args.aug_mode}-{args.image_size}_codes/{y[i]}/{index[i]}.npy', x[i])
        
        if is_main_process():
            pbar.update(1)
    if is_main_process():
        pbar.close()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-name", type=str, choices=['llamagen', 'maskgit'], default='llamagen')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="path to model config", default='configs/tokenizer.yaml')
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=256)
    parser.add_argument("--aug-mode", type=str, choices=['ten-crop', 'adm'], default='ten-crop')
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)