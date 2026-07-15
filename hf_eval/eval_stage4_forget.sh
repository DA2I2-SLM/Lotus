#!/bin/bash
#SBATCH --job-name=stage4_forget
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s
module load Anaconda3; source activate; conda activate opentslm
cd /home/$(whoami)/FoundationalTS
python -u evaluation/cot/eval_cot_phase2.py \
  --ckpt "/scratch/s223540177/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/stage4_sleep_cot_from_pretrained_nofreq/checkpoints/best_model.pt" \
  --llm_id google/gemma-3-270m --lora_r 4 --lora_alpha 8 \
  --tasks har sleep --max_samples 1000 \
  --out_dir /scratch/$(whoami)/FoundationalTS/eval_results/stage4_sleep_on_har_sleep
echo "=== stage4 forgetting eval done ==="
