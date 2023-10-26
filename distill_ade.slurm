#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/CIRKD/outs/ade20k/b2_b1_lr_.0001_batch_6_decay_1_iters_task_lambda_0.25.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=ADE20KB2
#SBATCH --constraint=gmem12
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 -m torch.distributed.launch --nproc_per_node=2 \
    train_kd.py \
    --lambda-kd 1.0 \
    --data /home/c3-0/datasets/ade/ADEChallengeData2016/ \
    --save-dir /home/aaryang/experiments/CIRKD/checkpoints/kd_b2_b1_pretrained_ade20k_.0001_6b_1_decay_task_0.25/ \
    --log-dir /home/aaryang/experiments/CIRKD/logs/ade20k/pretrained/ \
    --teacher-model b2 \
    --student-model b1 \
    --teacher-weights-path /home/aaryang/experiments/CIRKD/model_weights/b2_ade20k.pt \
    --student-weights-path /home/aaryang/experiments/CIRKD/model_weights/b1_ade20k.pt \
    --batch-size 6 \
    --max-iterations 35000 \
    --dataset cityscapes \
    --val-per-iters 200  \
    --pretrained-student True \
    --lr 0.0001 \
    --lr-decay-iterations 1  \
    --save-per-iters 10000 \
    --task-lambda 0.25 \
    --val-path /home/c3-0/datasets/ade/ADEChallengeData2016/
    

