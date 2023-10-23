#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/CIRKD/outs/pretrained_lr_.0001_batch_4_decay_1_iters_task_lambda_0.25_total_40000.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=lambda_task
#SBATCH --constraint=gmem16
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 -m torch.distributed.launch --nproc_per_node=2 \
    train_kd.py \
    --lambda-kd 1.0 \
    --data /home/c3-0/datasets/Cityscapes/ \
    --save-dir /home/aaryang/experiments/CIRKD/checkpoints/kd_l1_b0_pretrained_kld_cityscapes_.0001_4b_1_decay_task_0.25_40000/ \
    --log-dir /home/aaryang/experiments/CIRKD/logs/pretrained/normalized/ \
    --teacher-model l1 \
    --student-model b0 \
    --teacher-weights-path /home/aaryang/experiments/CIRKD/model_weights/l1.pt \
    --student-weights-path /home/aaryang/experiments/CIRKD/model_weights/b0.pt \
    --batch-size 8 \
    --max-iterations 40000 \
    --dataset cityscapes \
    --val-per-iters 200  \
    --pretrained-student True \
    --lr 0.0001 \
    --lr-decay-iterations 1  \
    --save-per-iters 10000 \
    --task-lambda 0.25 \
    --irregular-decay False \

    

