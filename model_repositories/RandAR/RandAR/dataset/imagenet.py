# Cloned from https://github.com/facebookresearch/deit/blob/main/datasets.py
# Modified from https://github.com/bfshi/AbSViT/blob/master/datasets.py

import os
import json
import numpy as np
import tarfile
from PIL import Image

import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class ImageTarDataset(data.Dataset):
    def __init__(self, tar_file, return_labels=False, transform=transforms.ToTensor()):
        '''
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        '''
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, 'r:') as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count('/') != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(tar_file, self.num))
        self.return_labels = return_labels
        self.transform = transform
        self.nb_classes = 0
  
    def _get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')
    
        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert('RGB')
        image = self.transform(image)
    
        if self.return_labels:
            category = self.categories[self._get_category_from_filename(
                self.tar_members[index].name)]
            return image, category, index
        return image, index


class INatLatentDataset(data.Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        categories_set = set()
        self.categories = sorted([int(i) for i in list(os.listdir(root_dir))])
        self.samples = []

        for tgt_class in self.categories:
            tgt_dir = os.path.join(root_dir, str(tgt_class))
            for root, _, fnames in sorted(os.walk(tgt_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, tgt_class)
                    self.samples.append(item)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(root_dir, self.num))
        self.transform = transform
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        sample = self.samples[index]
        latents = np.load(sample[0])
        latents = self.transform(latents) # 1 * aug_num * block_size

        # select one of the augmented crops
        aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
        latents = latents[:, aug_idx, :]
        label = sample[1]
        
        return latents, label, index
