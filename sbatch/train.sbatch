#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --job-name=attention_fine
#SBATCH --output=e2e6imges_baseline_train_rtx8000.out

module purge

singularity exec --nv --overlay /scratch/cc7287/environments/traj/habitat7.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /scratch/cc7287/env.sh; conda activate E2EParking; cd /vast/cc7287/trajectory/ParkWithUncertainty_develop_shunyu; python pl_train.py --model_path ckpt/exp_2025_4_6_14_2_39/last.ckpt"
            
