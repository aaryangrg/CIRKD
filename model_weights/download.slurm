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
URL_EFFICIENT_B2_CITY="https://drive.google.com/u/0/uc?id=1bwGjzVQOg_ygML8F9JhsIj-ntn-cuWmB&export=download"
URL_EFFICIENT_B1_CITY="https://drive.google.com/u/0/uc?id=1jNjLFtIUNvu5MwSupgFHLc-2kmFLiu67&export=download"
URL_EFFICIENT_B3_CITY="https://drive.google.com/u/0/uc?id=19aiy3qrKqx1n8zzy_ewYn4-Z3LM4bkn4&export=download"
URL_EFFICIENT_B2_ADE20K="https://drive.google.com/u/0/uc?id=1k5sWY6aJ1FCtMt4GRTZqSFlJ-u_TSHzc&export=download"
#gdown $URL_EFFICIENT_B0_CITYSCAPES
#gdown $URL_EFFICIENT_B1_ADE20K
#gdown $URL_EFFICIENT_L1_ADE20K
#gdown $URL_EFFICIENT_L1_CITYSCAPES
#gdown $URL_EFFICIENT_B0_CITY
#gdown $URL_EFFICIENT_B1_CITY
#gdown $URL_EFFICIENT_B3_CITY
gdown $URL_EFFICIENT_B2_ADE20K
echo "Model checkpoints downloaded"
