#!/bin/bash

#SBATCH --job-name=jupjob

#SBATCH --output=logs/gpu_job.out

#SBATCH --error=logs/gpu_job.err

#SBATCH --partition=gpu

#SBATCH --gres=gpu:2

#SBATCH --time=5-00:00:00 

#SBATCH --exclude=v100l-f-[01-06],v100-f-13

#SBATCH --qos=batch-short

#SBATCH --cpus-per-gpu=16

#SBATCH --nodelist=l40s-f-[01-04]

#SBATCH --ntasks=2


#SBATCH --mem-per-cpu=8g

 


module load Anaconda3
source activate
conda activate ts

jupyter notebook --port 8880 --ip 0.0.0.0 --no-browser
