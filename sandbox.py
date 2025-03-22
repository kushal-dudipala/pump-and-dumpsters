from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from pumpdumpsters.data_loader import load_and_preprocess_data
from pumpdumpsters.run_all import run_all_evaluation_metrics

import pandas as pd


# Load and preprocess data

print("Loading and preprocessing data for LSTM model...")
df = load_and_preprocess_data('test.csv') # update file path
print("Training LSTM model for next-close prediction...")
lstm_model, X_test, y_test = train_lstm_model(df, seq_len=3, epochs=10)

# Add the anomaly flags (e.g., z_score_anomaly, autoencoder_anomaly, lstm_anomaly) as extra columns
df = detect_lstm_anomalies(lstm_model, df, seq_len=3, percentile_threshold=95)

# evaluation metrics
run_all_evaluation_metrics(df)

print("Loading and preprocessing data for Autoencoder model...")
df = load_and_preprocess_data('test.csv') # update file path
print("Training Autoencoder for anomaly detection...")
autoencoder_model, X_test_autoencoder = train_autoencoder(df)

# Add the anomaly flags (e.g., z_score_anomaly, autoencoder_anomaly, lstm_anomaly) as extra columns
df = detect_autoencoder_anomalies(autoencoder_model, X_test_autoencoder, df)

# evaluation metrics
run_all_evaluation_metrics(df)







