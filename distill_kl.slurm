#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/CIRKD/outs/pretrained_cityscapes_normalized.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=EffViT_KD_full
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 -m torch.distributed.launch --nproc_per_node=2 \
    train_kd.py \
    --lambda-kd 1.0 \
    --data /home/c3-0/datasets/Cityscapes/ \
    --save-dir /home/aaryang/experiments/CIRKD/checkpoints/kd_l1_b0_pretrained_cross_kld_cityscapes_normalized/ \
    --log-dir /home/aaryang/experiments/CIRKD/logs/pretrained/normalized/ \
    --teacher_model l1 \
    --student_model b0 \
    --teacher_weights_path /home/aaryang/experiments/CIRKD/model_weights/l1.pt \
    --student_weights_path /home/aaryang/experiments/CIRKD/model_weights/b0.pt \
    --batch-size 16 \
    --max-iterations 20000 \
    --save-per-iters 2000 \
    --dataset cityscapes
 
