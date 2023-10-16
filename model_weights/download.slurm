#!/bin/bash
#SBATCH --output=/home/aaryang/experiments/CIRKD/model_weights/checkpoints.out
#SBATCH --account=group3
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1

# Set the paths where you want to save and unzip the downloaded files
# SAVE_PATH="/home/aaryang/experiments/efficientvit/assets/checkpoints/seg/cityscapes/b0.pt"
URL_EFFICIENT_B0="https://drive.google.com/u/0/uc?id=1iiy-LHpGIutXbPV-tJThAtz6jAgNcuxX&export=download"
# Create the save directory if it doesn't exist
# mkdir -p $SAVE_PATH

# Use wget to download the dataset files
gdown $URL_EFFICIENT_B0

# Unzip the downloaded files in the same directory

echo "Dataset downloaded and unzipped successfully to $SAVE_PATH"
