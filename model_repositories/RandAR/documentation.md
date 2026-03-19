# Getting Started

## 1. Installation

We require the most common dependencies:

* `Pytorch >= 2.1`
* `Accelerate` (use `==0.31.0` if you need the resume training feature)
* `einops`
* `omegaconf`
* `wandb` (for logging, can be set to `offline` mode)
* `tensorflow` (for FID evaluation)

## 2. Latent Code Extraction

To accelerate the training process, we use the pre-trained tokenizer from LLaMAGen or MaskGIT to extract the tokenized images. [[Our LLamAGEN Tokens]](https://huggingface.co/ziqipang/RandAR/resolve/main/imagenet-llamagen-adm-256_codes.tar), [[Our MaskGIT Tokens]](https://huggingface.co/ziqipang/RandAR/resolve/main/imagenet-maskgit-adm-256_codes.tar)

* Step 1: You can directly use our extracted latent codes without conducting the tokenization yourself.

* Step 2: If you want to extract the latent codes, please follow the steps below:

* Step 3: Download the pre-trained tokenizer from [LLaMAGen](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt) or [MaskGIT](https://huggingface.co/fun-research/TiTok/blob/main/maskgit-vqgan-imagenet-f16-256.bin). We use the tokenizers with downsampling factor of 16, by default.

* Step 4: Prepare the ImageNet dataset (I found [this script](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) helpful). For the convenience of moving ImageNet around from slow disk to fast computing nodes, I recommend you use `tar -cf` to compress the dataset into `train.tar` and `val.tar`. By default, I use the `ImageTarDataset` from [this file](./RandAR/dataset/imagenet.py) to handle them.

* Step 5: Run the following command to extract the tokenized images on the training sets. Several configurations:

  * `--data-path`: where you place the ImageNet training set (`train.tar`).
  * `--code-path`: where you want to save the extracted latent codes.
  * `--vq-ckpt`: the path to the pre-trained tokenizer.
  * `--config`: the path to the tokenizer config file ([LLaMAGen](./configs/tokenization/llamagen.yaml) or [MaskGIT](./configs/tokenization/maskgit.yaml)).
  * `--image-size`: the image size of the tokenized images.
  * `--aug-mode`: the augmentation mode. We use `adm`. `ten-crop` is the default choice of LLaMAGen and in our original papers, but it seems `adm` style only uses center crop and horizontal flipping and is better. Therefore, our re-implementation uses `adm` by default.

```bash
torchrun tools/extract_latent_codes.py \
    --data-path /tmp/ \
    --code-path /tmp/ \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --config configs/tokenization/llamagen.yaml \
    --image-size 256 \
    --aug-mode adm
```

## 3. Training

Our training script is `train_c2i.py`. The example command for training `RandAR-XL` is as below. Some of the critical configurations are as follows:

* `--config`: the path to the model config file ([`randar_xl_0.7b_llamagen.yaml`](./configs/randar/randar_xl_0.7b_llamagen.yaml)).
* `--data-path`: the path to the latent codes.
* `--vq-ckpt`: the path to the pre-trained tokenizer.
* `--results-dir`: the path to save the training checkpoints and results.
* `--disk-location`: the path to save the training checkpoints periodically to a permanent slow-speed disk. (Without specifying this, the option of periodically saving the checkpoints to a slow-speed disk will not be used.)

```
accelerate launch --mixed_precision=bf16 --multi_gpu \
    train_c2i.py --exp-name randar_0.7b_llamagen_360k \
    --config configs/randar/randar_xl_0.7b_llamagen.yaml \
    --data-path /tmp/imagenet-llamagen-adm-256_codes \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --results-dir /tmp \
    --disk-location /SLOW_DISK/training_ckpts \
```

### Scripts for All the Steps

Beginning from extracted tokens, we provide the scripts for launching the training from a plain compute node. Please checkout our [SLURM scripts](./slurm_scripts/) for a template.

### Where to Control the Optimization Parameters

We put all the modeling and optimization related hyper-parameters in the config files. Some of the most important ones are as below. They are mostly determined by the `global_batch_size: 1024` and a total of 300 epochs.

```yaml
accelerator:
  gradient_accumulation_steps: 1 # to support global_batch_size=1024
  mixed_precision: bf16
  log_with: wandb

optimizer:
  lr: 0.0004 # paired with global_batch_size=1024
  weight_decay: 0.05 # 5e-2
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  skip_grad_iter: 100
  skip_grad_norm: 10

lr_scheduler:
  type: cosine # you can also use constant
  warm_up_iters: 50000
  min_lr_ratio: 0.05
  num_cycles: 0.5

# training related parameters
max_iters: 360000 # paired with global_batch_size=1024, approximately 300 epochs steps
global_batch_size: 1024
```

NOTE: our paper uses a constant learning rate following LLaMAGen, but a cosine scheduler might be better. We are running experiments to verify this. Please stay tuned for an optimal default setting.

### Where to Control the Logging and Checkpointing

We put these into the `args` option of the `train_c2i.py` script. Some important configurations are:

* `--wandb-offline`: when debugging or using an offline machine, use this option to disable wandb remote syncing.
* `--log-every`: the frequency of logging.
* `--ckpt-every`: the frequency of saving checkpoints.
* `--visualize-every`: the frequency of visualizing the generated images.
* `--keep-last-k`: the number of checkpoints to keep.

## 4. Inference and Paralle Decoding

Given a trained model, such as 0.7B `RandAR-XL`, use the command like below to generate images. Some important configurations are:

* `--cfg-scales`: we use linear classifier-free guidance (CFG) by default. Specify the smallest and largest scale for CFG like "1.0,4.0" below. If you want to disable linear CFG, you can set it to "4.0,4.0" for a constant scale.
* `--num-inference-steps`: the number of inference steps, because we can use **paralle decoding**. For example, 256 steps means not using parallel decoding, while 88 steps means using parallel decoding.

Other than the above, you can also specify the following configurations:
* `--exp-name`: the name of the experiment.
* `--gpt-ckpt`: the path to the trained model checkpoint.
* `--vq-ckpt`: the path to the pre-trained tokenizer.
* `--config`: the path to the model config file.
* `--sample-dir`: the path to save the generated images.

```bash
torchrun sample_c2i.py \
    --exp-name sample_randar_0.7b_llamagen_360k \
    --gpt-ckpt /tmp/ckpt.safetensors \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --config configs/randar/randar_xl_0.7b.yaml \
    --cfg-scales 1.0,4.0 \
    --sample-dir ./samples \
    --num-inference-steps 88
```

## 5. Evaluation

Given a trained model, find the best CFG scale for FID evaluation. For efficiency, we search the best CFG scale at 0.2 intervals (`--cfg-scales-interval`) between 2.0 and 8.0 (`--cfg-scales-search`) using 10k samples (`--num-fid-samples-search`), then use the best CFG scale for the final 50k samples (`--num-fid-samples-final`) FID evaluation. The results will be saved into `--results-path` as a json file.

Please prepare the reference ImageNet dataset in adavnce for `--ref-path`. I downloaded it from [LLaMAGen](https://github.com/FoundationVision/LlamaGen/blob/main/evaluations/c2i/README.md), the $256\times 256$ reference for ImageNet is [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz).

```bash
torchrun tools/search_cfg_weights.py \
    --config configs/randar/randar_l_0.3b.yaml \
    --exp-name randar_0.3b_360k_llamagen \
    --gpt-ckpt /tmp/randar_0.3b_llamagen_360k.safetensors \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --per-proc-batch-size 128 \
    --num-fid-samples-search 10000 \
    --num-fid-samples-final 50000 \
    --cfg-scales-interval 0.2 \
    --cfg-scales-search 2.0,8.0 \
    --results-path ./results \
    --ref-path /tmp/VIRTUAL_imagenet256_labeled.npz \
    --sample-dir /tmp \
    --num-inference-steps 88
```

## 6. Diverse Zero-shot Applications

### Resolution Extrapolation

I know that this is one of the most interesting applications shown in RandAR. To try it out, just use the following command to generate $512\times 512$ images.

```bash
torchrun tools/search_cfg_weights.py \
    --config configs/randar/randar_xl_0.7b.yaml \
    --exp-name randar_0.7b_360k_llamagen_resolution_extrapolation \
    --gpt-ckpt /tmp/randar_0.7b_llamagen_360k.safetensors \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --per-proc-batch-size 8 \
    --cfg-scales 3.0,3.0 \
    --spatial-cfg-scales 2.5,2.5 \
    --num-inference-steps 1024 \
    --debug
  ```

  Since I am re-training the model, I found the behaviors of the existing checkpoints are not as stable as the ones used in the paper, probably due to a different learning rate schedule and break points during the training. I am trying to train a model following the original paper's settings. But I hope the current implementations can give everyone a good sense about our algorithms.