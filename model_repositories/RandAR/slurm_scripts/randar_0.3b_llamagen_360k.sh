#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4 
#SBATCH --account=ACCOUNT
#SBATCH --job-name=randar_0.3b_llamagen_360k
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=EMAIL
#SBATCH -o ./slurm_logs/randar_0.3b_llamagen_360k.out

# Extracted latent codes
# The following moves the data to the faster compute nodes
cp /work/nvme/bbsg/ziqip2/Datasets/imagenet/imagenet-llamagen-adm-256_codes.tar /tmp
cd /tmp
tar -xf imagenet-llamagen-adm-256_codes.tar
echo "Extracted latent codes"

# activate conda environment
source ~/.bashrc
conda activate /u/ziqip2/conda_envs/randar

# Run training
# Move the tokenizer to the faster compute nodes
cd /projects/RandAR
cp ../vq_ds16_c2i.pt /tmp

echo "Launching"

# Automatically handle resume training
# My cluster is not stable, so I periodically move the checkpoints to slow disk
# The following moves the checkpoints to the faster compute nodes
mkdir /tmp/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004/
mkdir /tmp/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004/checkpoints/
cp -r /work/hdd/bcnt/ziqip2/ARGen/training_ckpts/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004/* /tmp/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004/checkpoints/

accelerate launch --mixed_precision=bf16 --multi_gpu \
    train_c2i.py --exp-name randar_0.3b_llamagen_360k \
    --config configs/randar/randar_l_0.3b_llamagen.yaml \
    --data-path /tmp/imagenet-llamagen-adm-256_codes \
    --vq-ckpt /tmp/vq_ds16_c2i.pt --no-compile \
    --results-dir /tmp \
    --disk-location /work/hdd/bcnt/ziqip2/ARGen/training_ckpts