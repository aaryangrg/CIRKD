#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/CIRKD/outs/full_cityscapes_normalized.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=EffViT_KD_half
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 -m torch.distributed.launch --nproc_per_node=1 \
    train_kd.py \
    --lambda-kd 1.0 \
    --data /home/c3-0/datasets/Cityscapes/ \
    --save-dir /home/aaryang/experiments/CIRKD/checkpoints/kd_l1_b0_cross_kld_cityscapes_normalized/ \
    --log-dir /home/aaryang/experiments/CIRKD/logs/scratch/normalized/ \
    --teacher_model l1 \
    --student_model b0 \
    --teacher_weights_path /home/aaryang/experiments/CIRKD/model_weights/l1_cityscapes.pt \
    --batch-size 8 \
    --max-iterations 20000 \
    --save-per-iters 2000 \
    --dataset cityscapes
 
