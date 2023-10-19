#!/bin/bash
#SBATCH --output=/home/aaryang/experiments/CIRKD/model_weights/checkpoints.out
#SBATCH --account=group3
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --job-name=checkpoints
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

URL_EFFICIENT_B0_CITYSCAPES="https://drive.google.com/u/0/uc?id=1Ix1Dh3xlpaf0Wzh01Xmo-hAYkoXt1EAD&export=download"
URL_EFFICIENT_B1_ADE20K="https://drive.google.com/u/0/uc?id=13YNtEJ-mRnAhu0fIs2EnAP-3TmSneRAC&export=download"
URL_EFFICIENT_L1_ADE20K="https://drive.google.com/u/0/uc?id=1ErosNnztXT3ncrppWWPEMp5pag5WE-Vy&export=download"
URL_EFFICIENT_L1_CITYSCAPES="https://drive.google.com/u/0/uc?id=1iiy-LHpGIutXbPV-tJThAtz6jAgNcuxX&export=download"
URL_EFFICIENT_B0_CITY="https://drive.google.com/u/0/uc?id=1Ix1Dh3xlpaf0Wzh01Xmo-hAYkoXt1EAD&export=download"

#gdown $URL_EFFICIENT_B0_CITYSCAPES
#gdown $URL_EFFICIENT_B1_ADE20K
#gdown $URL_EFFICIENT_L1_ADE20K
gdown $URL_EFFICIENT_L1_CITYSCAPES
gdown $URL_EFFICIENT_B0_CITY

echo "Model checkpoints downloaded"
