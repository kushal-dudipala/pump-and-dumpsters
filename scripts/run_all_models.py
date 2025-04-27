'''
Trains all models using optimal hyperparameters.
'''
#data = os.path.join(project_root, "data", "processed", "merged_cryptos.csv")

import os
import sys
import subprocess
import traceback


# Setup 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import model components
from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from models.hybrid_cnn_lstm import train_hybrid_cnn_lstm_model, detect_hybrid_anomalies
from prepreprocessing.data_loader import load_and_preprocess_data
from pumpdumpsters.run_all import run_all_evaluation_metrics

def train_all_models():
    """Train all models on coin_Aave.csv and detect anomalies."""
    try:
        print("\n=== LOADING DATA ===")
        data = os.path.join(project_root, "data", "processed", "merged_cryptos.csv")
        df = load_and_preprocess_data(filename=data)

        # Define optimal hyperparameters
        lstm_params = {
            "seq_len": 30,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "test_size": 0.2
        }
        cnn_params = {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "test_size": 0.2
        }
        hybrid_params = {
            "seq_len": 30,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "cnn_filters": 64,
            "cnn_kernel_size": 3,
            "lstm_units": 64,
            "test_size": 0.2
        }

        # --- LSTM ---
        print("\n=== TRAINING LSTM MODEL ===")
        lstm_model, X_test_lstm, y_test_lstm = train_lstm_model(df.copy(), **lstm_params)
        df = detect_lstm_anomalies(lstm_model, df, seq_len=lstm_params["seq_len"], threshold=95)
        print("LSTM anomaly detection complete.")

        # --- CNN Autoencoder ---
        print("\n=== TRAINING CNN AUTOENCODER MODEL ===")
        autoencoder_model, X_test_autoencoder = train_autoencoder(df.copy(), **cnn_params)
        df = detect_autoencoder_anomalies(autoencoder_model, X_test_autoencoder, df)
        print("Autoencoder anomaly detection complete.")

        # --- HYBRID CNN-LSTM ---
        print("\n=== TRAINING HYBRID CNN-LSTM MODEL ===")
        hybrid_model, X_test_hybrid, y_test_hybrid = train_hybrid_cnn_lstm_model(df.copy(), **hybrid_params)
        df = detect_hybrid_anomalies(hybrid_model, df, seq_len=hybrid_params["seq_len"], threshold=95)
        print("Hybrid CNN-LSTM anomaly detection complete.")

        # --- EVALUATE ---
        print("\n=== RUNNING EVALUATION METRICS ===")
        evaluation_results = run_all_evaluation_metrics(df, project_root)
        print("\nEvaluation Results:")
        print(evaluation_results)

        print("\n=== ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY ===")

    except Exception as e:
        print("\n=== ERROR ===")
        print(f"Error in full model pipeline: {str(e)}")
        traceback.print_exc()



train_all_models()
