#!/bin/bash
#SBATCH --output=/home/aaryang/datasets/down.out
#SBATCH --account=group3
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1

cd /home/aaryang/datasets
URL_ADE20K="https://groups.csail.mit.edu/vision/datasets/ADE20K/syml/aaryangrg_58c9fc3a.zip"
wget $URL_ADE20K
echo "Dataset Downloaded"
unzip aaryangrg_58c9fc3a.zip
echo "Dataset Unzipped"

