import math
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torchvision import datasets


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(
    pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, g=None
):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = torch.randint(
        min_smaller_dim_size, max_smaller_dim_size + 1, (1,), generator=g
    ).item()

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = torch.randint(0, arr.shape[0] - image_size + 1, (1,), generator=g).item()
    crop_x = torch.randint(0, arr.shape[1] - image_size + 1, (1,), generator=g).item()
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def sample_imagenet(DATASET_PATH, IMAGENET_PATH, OUTPUT_PATH, N_SAMPLES=None):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    class_to_idx = datasets.ImageFolder(
        IMAGENET_PATH
    ).class_to_idx  # {'n01440764': 0, ...}
    dataset = datasets.ImageFolder(DATASET_PATH)

    samples_per_class = {}

    for path, class_idx in dataset.samples:
        class_name = path.split("/")[-2]
        class_idx = class_to_idx[class_name]
        if class_idx not in samples_per_class:
            samples_per_class[class_idx] = [path]
        elif N_SAMPLES is None or len(samples_per_class[class_idx]) < N_SAMPLES:
            samples_per_class[class_idx].append(path)

    class_idxs = []
    for class_idx, img_paths in samples_per_class.items():
        class_idxs.append(class_idx)
        for i in range(len(img_paths)):
            dst = os.path.join(OUTPUT_PATH, f"{class_idx}_{i}.JPEG")
            shutil.copy(img_paths[i], dst)

    return class_idxs


def calc_statistics(likelyhoods, targets, n_trials_acc, classes):
    len_dataset, n_trials, n_class = likelyhoods.shape
    n_averaging = n_trials // n_trials_acc

    accuracy_list_0 = []
    for i in range(n_averaging):
        likelyhoods_ = likelyhoods[
            :, i * n_trials_acc : (i + 1) * n_trials_acc
        ]  # (len_dataset, n_trials_acc, n_class)
        preds = classes[likelyhoods_.mean(dim=1).argmax(dim=-1)]
        accuracy_list_0.append((preds == targets).float().mean().item())

    accuracy_list_1 = []
    for i in range(n_averaging):
        likelyhoods_ = likelyhoods[
            :, i * n_trials_acc : (i + 1) * n_trials_acc
        ]  # (len_dataset, n_trials_acc, n_class)
        preds = classes[torch.softmax(likelyhoods_, dim=-1).mean(dim=1).argmax(dim=-1)]
        accuracy_list_1.append((preds == targets).float().mean().item())

    accuracy_list_2 = []
    for i in range(n_averaging):
        likelyhoods_ = likelyhoods[
            :, i * n_trials_acc : (i + 1) * n_trials_acc
        ]  # (len_dataset, n_trials_acc, n_class)
        preds = classes[torch.logsumexp(likelyhoods_, dim=1).argmax(dim=-1)]
        accuracy_list_2.append((preds == targets).float().mean().item())

    return (
        np.array(accuracy_list_0),
        np.array(accuracy_list_1),
        np.array(accuracy_list_2),
    )
