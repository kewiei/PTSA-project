#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=90GB
#SBATCH --job-name=MLPReturnAnaLyse
#SBATCH --mail-type=END
#SBATCH --mail-user=tz1264@nyu.edu
#SBATCH --output=slurm_MLPReturnAnaLyse_%j.out

cd /home/tz1264
module purge
source activate xgb
python MLPReturnAnalyse.py