#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=90GB
#SBATCH --job-name=CPULogReg
#SBATCH --mail-type=END
#SBATCH --mail-user=tz1264@nyu.edu
#SBATCH --output=slurm_CPULogReg_%j.out

cd /home/tz1264
module purge
source activate xgb
python LogRegression.py