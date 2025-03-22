#!/bin/bash
#SBATCH --job-name=pump-and-dump-test       # Job name
#SBATCH --output=testing%j.out              # Standard output file
#SBATCH --error=testing%j.err               # Error file
#SBATCH --partition=ice-gpu                 # Partition name (check with 'sinfo' if needed)
#SBATCH -N1 --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=4                   # Request 4 CPU cores
#SBATCH --mem=16G                           # Request 16GB RAM
#SBATCH --time=01:00:00                     # Max job runtime (hh:mm:ss)
#SBATCH --mail-type=END,FAIL                # Email notification (optional)
#SBATCH --mail-user=kdudipala3@gatech.edu   # Replace with your email

# Load necessary modules (modify as per your HPC environment)
module load anaconda3 # If Conda is available
source activate my_env # Activate your Conda environment

# Debugging: Print the GPU details
echo "Running on GPU:"

# Run the DeepSeek Python script
python scripts/train_model.py
