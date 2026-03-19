# Open Source 🎲 RandAR: Decoder-only Autoregressive Visual Generation in Random Orders

[[`Project Page`](https://rand-ar.github.io/)] [[`arXiv`](https://arxiv.org/abs/2412.01827)] [[`HuggingFace`](https://huggingface.co/ziqipang/RandAR)]

[![arXiv](https://img.shields.io/badge/arXiv-2412.01827-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2412.01827)
[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://rand-ar.github.io/) 
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat&logo=HuggingFace&logoColor=blue)](https://huggingface.co/ziqipang/RandAR)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## Overview

Ever thinking about what is the prerequisite for a visual model achieving the impact of GPT in language? The prequisite should be its ability of zero-shot generalization to various applications, prompts, etc. Our RandAR is one of the attempts towards this objective.

🎲 **RandAR** is a decoder-only AR model generating image tokens in arbitrary orders. 

🚀 **RandAR** supports parallel-decoding without additional fine-tuning and brings 2.5 $\times$ acceleration for AR generation.

🛠️ **RandAR** unlocks new capabilities for causal GPT-style transformers: inpainting, outpainting, zero-shot resolution extrapolation, and bi-directional feature encoding.

<img src="imgs/teaser.png" alt="teaser" width="100%">

## News

- [12/29/2024] 🎉 We release example checkpoints for RandAR. We are continuing to train more models and release the code supporting the diverse zero-shot tasks by 1/09/2025.

- [12/09/2024] 🎉 The initial code is released, including the tokenization/modeling/training pipeline. I found that augmentation & tokenization different from the LLaMAGEN's designs are better for FID. From the current speed of training, I expect to release model checkpoints and verified training/eval scripts before 12/18/2024.

- [12/02/2024] 📋 I am trying my best to re-implement the code and re-train the model as soon as I can. I plan to release the code before 12/09/2024 and the models afterwards. I am going to make my clusters running so fiecely that they will warm up the whole Illinois during this winter. 🔥🔥🔥

- [12/02/2024] 🎉 The paper appears on Arxiv.

## Getting Started

Checkout our documentation [DOCUMENTATION.md](documentation.md) for more details.

## Pre-trained Models

We have tried slightly modified (1) tokenizer: using either MaskGIT's or LLaMAGen's tokenizer, (2) learning rate schedule: using either cosine or linear schedule, for training RandAR with the hope that they could improve the performance. Their performance is slightly different from the paper's numbers, and we will release the checkpoints following the paper's numbers soon. All the checkpoints are available on [HuggingFace](https://huggingface.co/ziqipang/RandAR).

We would like to highlight two observations:

- Using MaskGIT's tokenizer improves the FID, because of its smaller vocabulary.
- Using cosine or linear learning rate does not show significant performance difference at 0.7B model size.

| Model | Param | Tokenizer | LR Schedule | Optimal CFG | FID | IS | Precision | Recall | Training Finished |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandAR-L (Paper) | 0.3B | LLaMAGen | Linear | - | 2.55 | 288 | 0.82 | 0.58 | N |
| [RandAR-L](https://huggingface.co/ziqipang/RandAR/blob/main/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors) | 0.3B | LLaMAGen | Cosine | 3.4 | 2.65 | 249 | 0.82 | 0.56 | Y |
| [RandAR-L](https://huggingface.co/ziqipang/RandAR/blob/main/randar_0.3b_maskgit_360k_bs_1024_lr_0.0004.safetensors) | 0.3B | MaskGIT | Cosine | 4.0 | 2.47 | 271 | 0.84 | 0.54 | Y |
| RandAR-XL (Paper) | 0.7B | LLaMAGen | Linear | - | 2.25 | 318 | 0.80 | 0.60 | N |
| [RandAR-XL](https://huggingface.co/ziqipang/RandAR/blob/main/randar_0.7b_llamagen_360k_bs_1024_lr_0.0004.safetensors) | 0.7B | LLaMAGen | Cosine | 4.0 | 2.27 | 275 | 0.81 | 0.59 | Y |
| RandAR-XL | 0.7B | MaskGIT | Cosine | - | - | - | - | - | N |

## Related and Follow-up Works

After the release of RandAR, we are thrilled to see the impressive works highly related to or extending RandAR. Please let me know if you wish more people could see your work.

- [Autoregressive Image Generation with Randomized Parallel Decoding](https://github.com/hp-l33/ARPG). Haopeng Li, Jinyue Yang, Guoqi Li, Huan Wang.

- [Parallelized Autoregressive Visual Generation](https://github.com/YuqingWang1029/PAR). Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, Xihui Liu. CVPR 2025

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pang2024randar,
    title={RandAR: Decoder-only Autoregressive Visual Generation in Random Orders},
    author={Pang, Ziqi and Zhang, Tianyuan and Luan, Fujun and Man, Yunze and Tan, Hao and Zhang, Kai and Freeman, William T. and Wang, Yu-Xiong},
    journal={arXiv preprint arXiv:2412.01827},
    year={2024}
}
```

## Acknowledgement

Thank you to the open-source community for their explorations on autoregressive generation, especially [LLaMAGen](https://github.com/FoundationVision/LlamaGen).
