#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=GPUmlp
#SBATCH --mail-type=END
#SBATCH --mail-user=tz1264@nyu.edu
#SBATCH --output=slurm_%j.out

cd /home/tz1264
module load cudnn/10.0v7.6.2.24
source activate test
python /home/tz1264/mlp_imple.py