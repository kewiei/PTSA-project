#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=CPUGP
#SBATCH --mail-type=END
#SBATCH --mail-user=tz1264@nyu.edu
#SBATCH --output=/beegfs/tz1264/slurm_CPUGP_%j.out

cd /home/tz1264
module purge
source activate xgb
python GP.py