import argparse
import itertools
import subprocess
import traceback
from utils.save_best_hyperparams import save_best_hyperparams

parser = argparse.ArgumentParser(description="Run hyperparameter sweep for a specified model.")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    choices=["cnn", "lstm", "hybrid"],
    help="The type of model to run the hyperparameter sweep for (cnn, lstm, hybrid)."
)
args = parser.parse_args()
model = args.model

try:
    print(f"\n=== HYPERPARAMETER SWEEP: {model.upper()} MODEL ===")

    # Define hyperparameter grid
    if model == "lstm":
        seq_lens = [3, 5, 7]
        epochs_list = [50]
        lstm_units = [32, 64, 128]
        learning_rates = [0.001, 0.01]
        param_grid = list(itertools.product(seq_lens, epochs_list, lstm_units, learning_rates))
    elif model == "cnn":
        epochs_list = [50]
        cnn_filters = [32, 64, 128]
        kernel_sizes = [3, 5]
        learning_rates = [0.001, 0.01]
        param_grid = list(itertools.product(epochs_list, cnn_filters, kernel_sizes, learning_rates))
    elif model == "hybrid":
        seq_lens = [3, 5, 7]
        epochs_list = [50]
        cnn_filters = [32, 64, 128]
        lstm_units = [32, 64, 128]
        learning_rates = [0.001, 0.01]
        param_grid = list(itertools.product(seq_lens, epochs_list, cnn_filters, lstm_units, learning_rates))

    # Track best model and results
    best_score = -float("inf")
    best_params = None
    all_results = []

    for idx, params in enumerate(param_grid):
        print(f"\n--- Running combination {idx + 1}/{len(param_grid)} ---")
        print(f"Parameters: {params}")

        # Construct CLI command
        cmd = ["python", "/Users/kushaldudipala/codebase/CS4644/project/pump-and-dumpsters/scripts/train_models.py", "--model", model]
        if model == "lstm":
            seq_len, epochs, units, lr = params
            cmd += ["--seq_len", str(seq_len), "--epochs", str(epochs), "--lstm_units", str(units), "--learning_rate", str(lr)]
        elif model == "cnn":
            epochs, filters, kernel_size, lr = params
            cmd += ["--epochs", str(epochs), "--cnn_filters", str(filters), "--cnn_kernel_size", str(kernel_size), "--learning_rate", str(lr)]
        elif model == "hybrid":
            seq_len, epochs, filters, units, lr = params
            cmd += ["--seq_len", str(seq_len), "--epochs", str(epochs), "--cnn_filters", str(filters), "--lstm_units", str(units), "--learning_rate", str(lr)]

        # Run the command and capture output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            # Parse F1 score from the output (assuming it's printed in the output)
            for line in result.stdout.splitlines():
                if "F1 Score:" in line:
                    score = float(line.split(":")[1].strip())
                    break
            else:
                score = 0  # Default if F1 score not found
        except subprocess.CalledProcessError as e:
            print(f"Error during training: {e.stderr}")
            score = 0

        all_results.append((*params, score))

        if score > best_score:
            best_score = score
            best_params = params

    print("\n=== HYPERPARAMETER SWEEP COMPLETE ===")
    print(f"Best Params: {best_params}")
    print(f"Best Score (F1): {best_score}")
    save_best_hyperparams(model, best_params, best_score)

except Exception as e:
    print("\n=== ERROR DURING SWEEP ===")
    print(f"Error in hyperparameter sweep: {str(e)}")
    traceback.print_exc()
