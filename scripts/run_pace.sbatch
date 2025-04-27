#!/bin/bash
#SBATCH --job-name=pump-and-dump-sweep          # Job name
#SBATCH --output=sweep%j.out                    # Standard output file
#SBATCH --error=sweep%j.err                     # Error file
#SBATCH --partition=ice-gpu                     # Partition name (check with 'sinfo' if needed)
#SBATCH -N1 --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=4                       # Request 4 CPU cores
#SBATCH --mem=16G                               # Request 16GB RAM
#SBATCH --time=03:00:00                         # Max job runtime (hh:mm:ss)
#SBATCH --mail-type=END,FAIL                    # Email notification (optional)
#SBATCH --mail-user=kdudipala3@gatech.edu       # Replace with your email

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pumpdumpsters-env
python3 run_all_models.py 