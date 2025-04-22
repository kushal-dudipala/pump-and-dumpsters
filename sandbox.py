import os
import pandas as pd
import numpy as np
import tensorflow as tf
import traceback

# Make TensorFlow less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import our models
from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from models.hybrid_cnn_lstm import train_hybrid_cnn_lstm_model, detect_hybrid_anomalies
from pumpdumpsters.data_loader import load_and_preprocess_data
from pumpdumpsters.run_all import run_all_evaluation_metrics

def run_model_pipeline():
    """Run the complete anomaly detection pipeline using all three models."""
    try:
        # Load and preprocess data (will automatically use sample data)
        print("\n=== LOADING DATA ===")
        print("Loading and preprocessing data for models...")
        df = load_and_preprocess_data()
        
        # Reduce epochs for faster testing
        test_epochs = 5
        seq_len = 3
        
        # LSTM Model
        print("\n=== LSTM MODEL ===")
        print(f"Training LSTM model for next-close prediction (epochs={test_epochs})...")
        lstm_model, X_test_lstm, y_test_lstm = train_lstm_model(df, seq_len=seq_len, epochs=test_epochs)

        # Add the LSTM anomaly flags
        df = detect_lstm_anomalies(lstm_model, df, seq_len=seq_len, threshold=95)
        print("LSTM anomaly detection complete")

        # Autoencoder Model
        print("\n=== AUTOENCODER MODEL ===")
        print(f"Training Autoencoder for anomaly detection (epochs={test_epochs})...")
        autoencoder_model, X_test_autoencoder = train_autoencoder(df, epochs=test_epochs)

        # Add the Autoencoder anomaly flags
        df = detect_autoencoder_anomalies(autoencoder_model, X_test_autoencoder, df)
        print("Autoencoder anomaly detection complete")

        # Hybrid CNN-LSTM Model
        print("\n=== HYBRID CNN-LSTM MODEL ===")
        print(f"Training Hybrid CNN-LSTM model (epochs={test_epochs})...")
        hybrid_model, X_test_hybrid, y_test_hybrid = train_hybrid_cnn_lstm_model(df, seq_len=seq_len, epochs=test_epochs)

        # Add the Hybrid anomaly flags
        df = detect_hybrid_anomalies(hybrid_model, df, seq_len=seq_len, threshold=95)
        print("Hybrid CNN-LSTM anomaly detection complete")

        # Run evaluation metrics
        print("\n=== EVALUATION ===")
        print("Running evaluation metrics on all models...")
        evaluation_results = run_all_evaluation_metrics(df)
        print("Evaluation complete")
        
        print("\n=== SUCCESS ===")
        print("All models trained and evaluated successfully!")
        return True
    
    except Exception as e:
        print("\n=== ERROR ===")
        print(f"Error in model pipeline: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_model_pipeline()







