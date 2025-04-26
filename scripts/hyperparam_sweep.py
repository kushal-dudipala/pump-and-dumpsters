import argparse
import itertools
import numpy as np
import traceback

from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from models.hybrid_cnn_lstm import train_hybrid_cnn_lstm_model, detect_hybrid_anomalies
from pumpdumpsters.data_loader import load_and_preprocess_data
from pumpdumpsters.run_all import run_all_evaluation_metrics

parser = argparse.ArgumentParser(description="Run hyperparameter sweep for a specified model.")
parser.add_argument(
    "--model",   
    required=True,
    type=str,
    choices=["cnn", "lstm", "hybrid"],
    help="The type of model to run the hyperparameter sweep for (cnn, lstm, hybrid)."
)
args = parser.parse_args()
model = parser.parse_args().model


"""Hyperparameter sweep"""
try:
    print(f"\n=== HYPERPARAMETER SWEEP: {model.upper()} MODEL ===")
    
    # Define valid models and their corresponding functions
    model_dict = {
        "lstm": {
            "train": train_lstm_model,
            "detect": detect_lstm_anomalies
        },
        "cnn": {
            "train": train_autoencoder,  
            "detect": detect_autoencoder_anomalies
        },
        "hybrid": {
            "train": train_hybrid_cnn_lstm_model,
            "detect": detect_hybrid_anomalies
        }
    }

    if model not in model_dict:
        raise ValueError(f"Invalid model type '{model}'. Must be one of {list(model_dict.keys())}.")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Define hyperparameter grid
    if model == "lstm":
        seq_lens = [3, 5]
        epochs_list = [5, 10]
        lstm_units = [32, 64]
        param_grid = list(itertools.product(seq_lens, epochs_list, lstm_units))
    elif model == "cnn":
        epochs_list = [5, 10]
        cnn_filters = [32, 64]
        param_grid = list(itertools.product(epochs_list, cnn_filters))
    elif model == "hybrid":
        seq_lens = [3, 5]
        epochs_list = [5, 10]
        cnn_filters = [32, 64]
        lstm_units = [32, 64]
        param_grid = list(itertools.product(seq_lens, epochs_list, cnn_filters, lstm_units))

    # Track best model and results
    best_score = -np.inf
    best_params = None
    all_results = []

    for idx, params in enumerate(param_grid):
        print(f"\n--- Running combination {idx + 1}/{len(param_grid)} ---")
        print(f"Parameters: {params}")

        if model == "lstm":
            seq_len, epochs, units = params
            trained_model, X_test, y_test = model_dict[model]["train"](
                df,
                seq_len=seq_len,
                epochs=epochs,
                lstm_units=units
            )
            df_temp = model_dict[model]["detect"](trained_model, df.copy(), seq_len=seq_len, threshold=95)
            
        elif model == "cnn":
            epochs, filters = params
            trained_model, X_test, y_test = model_dict[model]["train"](
                df,
                epochs=epochs,
                cnn_filters=filters
            )
            df_temp = model_dict[model]["detect"](trained_model, df.copy(), threshold=95)
            
        elif model == "hybrid":
            seq_len, epochs, filters, units = params
            trained_model, X_test, y_test = model_dict[model]["train"](
                df,
                seq_len=seq_len,
                epochs=epochs,
                cnn_filters=filters,
                lstm_units=units
            )
            df_temp = model_dict[model]["detect"](trained_model, df.copy(), seq_len=seq_len, threshold=95)

        evaluation_result = run_all_evaluation_metrics(df_temp)
        score = evaluation_result.get(f"{model.capitalize()}_F1_score", 0)
        print(f"F1 Score: {score}")

        all_results.append((*params, score))

        if score > best_score:
            best_score = score
            best_params = params

    print("\n=== HYPERPARAMETER SWEEP COMPLETE ===")
    print(f"Best Params: {best_params}")
    print(f"Best Score (F1): {best_score}")
    
    output_of_sweeps = all_results, best_params
    

except Exception as e:
    print("\n=== ERROR DURING SWEEP ===")
    print(f"Error in hyperparameter sweep: {str(e)}")
    traceback.print_exc()
    output_of_sweeps =  None
    
def get_output():
    return output_of_sweeps

def get_best_params():
    return best_params


   
