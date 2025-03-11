from data_loader import load_and_preprocess_data
from eda import plot_price_trends, plot_volume_trends
from anomaly_detection import detect_anomalies_zscore, plot_zscore_anomalies
from deep_learning import train_autoencoder, detect_autoencoder_anomalies
from mean_reversion import apply_mean_reversion_strategy, plot_mean_reversion
from evaluation import evaluate_model
from utils import print_separator

# Load and preprocess data
print_separator()
print("Loading and preprocessing data...")
df = load_and_preprocess_data()

# Perform Exploratory Data Analysis
print_separator()
print("Performing EDA...")
plot_price_trends(df)
plot_volume_trends(df)

# Baseline Anomaly Detection
print_separator()
print("Detecting anomalies with Z-score...")
df = detect_anomalies_zscore(df)
plot_zscore_anomalies(df)

# Train Autoencoder
print_separator()
print("Training Autoencoder model...")
model, X_test = train_autoencoder(df)

# Autoencoder Anomaly Detection
print_separator()
print("Detecting anomalies using Autoencoder...")
df = detect_autoencoder_anomalies(model, X_test, df)

# Apply Mean Reversion Strategy
print_separator()
print("Applying mean reversion strategy...")
df = apply_mean_reversion_strategy(df)
plot_mean_reversion(df)

# Evaluate Model Performance
print_separator()
print("Evaluating Model Performance...")
evaluate_model(df)
