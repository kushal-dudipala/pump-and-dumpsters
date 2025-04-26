import sys
import os
import subprocess
import time
import traceback

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from models.hybrid_cnn_lstm import train_hybrid_cnn_lstm_model, detect_hybrid_anomalies
from prepreprocessing.data_loader import load_and_preprocess_data


def run_single_model_test(model_name):
    print(f"\n=== Testing {model_name.upper()} model ===")
    
    # Paths already set globally
    train_script = os.path.join(current_dir, "train_models.py")
    merged_data = os.path.join(project_root, "data", "processed", "merged_cryptos.csv")

    # Build the command
    cmd = [
        "python", train_script,
        "--model", model_name,
        "--epochs", "1",
        "--batch_size", "16",
        "--learning_rate", "0.001",
        "--seq_len", "30",             # For lstm and hybrid
        "--cnn_filters", "64",          # For hybrid
        "--cnn_kernel_size", "3",       # For hybrid
        "--lstm_units", "64",           # For lstm and hybrid
        "--filename", merged_data
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during {model_name.upper()} training:\n{e.stderr}")
    
    elapsed = time.time() - start_time
    print(f"=== {model_name.upper()} completed in {elapsed:.2f} seconds ===\n")


def main():
    models = ["lstm", "cnn", "hybrid"]
    for model in models:
        run_single_model_test(model)


if __name__ == "__main__":
    main()
