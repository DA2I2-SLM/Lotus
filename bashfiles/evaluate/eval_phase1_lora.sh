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
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name dominick   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/dominick --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name monash_fred_md   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/monash_fred_md --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name electricity_15min   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/electricity_15min --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name ercot   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/ercot --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name exchange_rate   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/exchange_rate --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name ushcn_daily   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/ushcn_daily --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name weatherbench_daily   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/weatherbench_daily --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name weatherbench_weekly   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/weatherbench_weekly --lora_r 16
python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/inference.py   --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_lora/checkpoints/periodic_checkpoint.pt   --dataset_name wiki_daily_100k   --llm_id google/gemma-3-270m   --device cuda   --batch_size 4   --patch_size 16   --out_dir ./phase1_ts_eval/wiki_daily_100k --lora_r 16

