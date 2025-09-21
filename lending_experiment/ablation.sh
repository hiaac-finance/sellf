#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=/home/giovani.valdrighi/fairrl/ablation.out
#SBATCH --error=/home/giovani.valdrighi/fairrl/ablation.err
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=g272455@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-24
source ~/miniconda3/bin/activate
conda activate fairrl
cd /home/giovani.valdrighi/fairrl/lending_experiment

# sellf
python ablation.py --env_name fico_equal --algorithm sellf --mu_type accuracy --seed_id $SLURM_ARRAY_TASK_ID






