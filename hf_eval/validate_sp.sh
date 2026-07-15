#!/bin/bash
#SBATCH --job-name=val_sp
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s
module load Anaconda3
source activate
conda activate opentslm
cd /home/$(whoami)/FoundationalTS
echo "### CONFIG A: encoder-fix + default attn (sdpa+bf16) ###"
python -u hf_eval/run_eval.py --repo_id OpenTSLM/gemma-3-270m-pt-sleep-sp \
  --tasks sleep --max_samples 24 --batch_size 1 --max_new_tokens 300 \
  --attn_impl default --out_dir /scratch/$(whoami)/FoundationalTS/hf_eval_results/_val_sp_sdpa
echo "### CONFIG B: encoder-fix + eager ###"
python -u hf_eval/run_eval.py --repo_id OpenTSLM/gemma-3-270m-pt-sleep-sp \
  --tasks sleep --max_samples 24 --batch_size 1 --max_new_tokens 300 \
  --attn_impl eager --llm_dtype bf16 --out_dir /scratch/$(whoami)/FoundationalTS/hf_eval_results/_val_sp_eager
echo "=== validate_sp done ==="
