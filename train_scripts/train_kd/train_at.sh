#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/efficientvit/cityscape.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi



python3 -m torch.distributed.launch --nproc_per_node=1 \
    train_kd.py \
    --lambda-kd 1.0 \
    --data /share/datasets/Cityscapes/ \
    --save-dir /home/aaryang/experiments/CIRKD/checkpoints/ \
    --log-dir /home/aaryang/experiments/CIRKD/logs/ \
    --batch-size 8
