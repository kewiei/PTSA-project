#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=CPUxgboost
#SBATCH --mail-type=END
#SBATCH --mail-user=tz1264@nyu.edu
#SBATCH --output=slurm_%j.out

cd /home/tz1264
module purge
source activate xgb
python /home/tz1264/xgboost_imple.py