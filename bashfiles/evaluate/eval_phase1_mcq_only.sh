#!/bin/bash
#SBATCH --job-name=tsqa-evaluation
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]
module load Anaconda3
source activate
conda activate opentslm
USER_NAME=$(whoami)

cd /home/${USER_NAME}/FoundationalTS
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name dominick   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/dominick
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name monash_fred_md   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/monash_fred_md
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name electricity_15min   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/electricity_15min
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name ercot   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/ercot
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name exchange_rate   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/exchange_rate
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name ushcn_daily   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/ushcn_daily
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name weatherbench_daily   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/weatherbench_daily
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name weatherbench_weekly   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/weatherbench_weekly
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/nope_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt   --dataset_name wiki_daily_100k   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./stage1_mcq_nope/wiki_daily_100k