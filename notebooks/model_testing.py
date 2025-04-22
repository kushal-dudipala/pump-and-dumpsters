#!/usr/bin/env python3
# Cryptocurrency Anomaly Detection: Model Testing
#
# This script performs comprehensive testing and comparison of three anomaly
# detection models (LSTM, Autoencoder, and Hybrid CNN-LSTM) on cryptocurrency data.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from datetime import datetime

# Add project root to path
sys.path.append('..')

# Import project modules
try:
    from models.lstm_model import LSTMModel
    from models.auto_encoder import AutoEncoder
    from models.hybrid_cnn_lstm import HybridCNNLSTM
    from models.utils import load_data, preprocess_data, split_data
except ImportError as e:
    print(f"Warning: {e}")
    print("Using mock model classes for demonstration")
    
    # Define mock model classes for testing
    class BaseMockModel:
        def __init__(self, input_shape):
            self.input_shape = input_shape
            print(f"Initialized mock model with input shape {input_shape}")
            
        def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=0):
            print(f"Training mock model with {len(X_train)} samples for {epochs} epochs")
            # Simulate training time
            return None
            
        def predict(self, X_test):
            print(f"Generating predictions for {len(X_test)} samples")
            # Return random predictions
            return np.random.randint(0, 2, size=len(X_test))
    
    # Mock classes for each model type
    class LSTMModel(BaseMockModel): pass
    class AutoEncoder(BaseMockModel): pass
    class HybridCNNLSTM(BaseMockModel): pass
    
    # Mock utility functions
    def load_data(file_path):
        print(f"Mock load_data called with {file_path}")
        return None
        
    def preprocess_data(df, feature_columns=None, target_column=None):
        print(f"Mock preprocess_data called with {len(df)} rows")
        # Return dummy data
        X = np.random.rand(len(df), len(feature_columns))
        y = np.random.randint(0, 2, size=len(df))
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        return X, y, scaler
        
    def split_data(X, y, sequence_length):
        print(f"Mock split_data called with sequence length {sequence_length}")
        # Return dummy sequences
        n_samples = len(X) - sequence_length + 1
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X_seq = np.random.rand(n_samples, sequence_length, n_features)
        y_seq = np.random.randint(0, 2, size=n_samples)
        return X_seq, y_seq

# Set up plotting
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def main():
    """Main function for model testing and comparison"""
    print("Cryptocurrency Anomaly Detection: Model Testing")
    print("=" * 60)
    
    # Configuration
    models_to_test = ['lstm', 'autoencoder', 'hybrid']
    test_size = 0.2
    random_state = 42
    
    # Create plots directory if it doesn't exist
    os.makedirs("../plots", exist_ok=True)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    if data is None:
        print("Data preparation failed. Exiting.")
        return
    
    # Split features and target
    X_train, X_test, y_train, y_test, scaler, date_test = data
    
    # Test individual models
    results = {}
    predictions = {}
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()} model")
        print("-" * 60)
        
        # Train and evaluate model
        y_pred, model = test_model(model_name, X_train, X_test, y_train, scaler)
        
        # Store predictions
        predictions[model_name] = y_pred
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred)
        results[model_name] = metrics
        
        # Print metrics
        print(f"\n{model_name.upper()} Model Performance:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
    
    # Compare models
    compare_models(results, predictions, y_test, date_test)
    
    # Save results
    save_results(results)
    
    print("\nModel testing completed!")

def load_and_prepare_data():
    """Load and prepare data for model testing"""
    print("\n1. Loading and Preparing Data")
    print("-" * 60)
    
    try:
        # Try to load sample data
        data_path = "../data/raw/sample_bitcoin.csv"
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found.")
            print("Using synthetic data for demonstration.")
            
            # Create synthetic data
            days = 500
            dates = pd.date_range(start='2020-01-01', periods=days)
            
            # Generate price with some trends and anomalies
            base_price = 10000 + np.cumsum(np.random.normal(0, 200, days))
            trend = np.linspace(0, 5000, days)
            
            # Add seasonality
            seasonality = 1000 * np.sin(np.linspace(0, 10 * np.pi, days))
            
            # Add anomalies (abrupt changes)
            anomalies = np.zeros(days)
            anomaly_indices = np.random.choice(range(days), size=15, replace=False)
            for idx in anomaly_indices:
                if idx < days - 1:
                    anomalies[idx] = np.random.choice([-1, 1]) * np.random.uniform(1000, 3000)
            
            # Combine components
            price = base_price + trend + seasonality + anomalies
            
            # Create dataframe
            df = pd.DataFrame({
                'date': dates,
                'price': price,
                'volume': np.random.lognormal(10, 1, days) * 100
            })
            
            # Label anomalies based on large price changes
            df['price_change'] = df['price'].diff().abs()
            threshold = df['price_change'].mean() + 2.5 * df['price_change'].std()
            df['anomaly'] = (df['price_change'] > threshold).astype(int)
            
            print(f"Created synthetic data with {len(df)} entries")
        else:
            # Load real data
            df = pd.read_csv(data_path)
            
            # Rename columns to standardized format
            if 'Date' in df.columns:
                df = df.rename(columns={
                    'Date': 'date',
                    'Close': 'price',
                    'Volume': 'volume',
                    'Marketcap': 'market_cap'
                })
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Check if anomaly column exists, otherwise create it
            if 'anomaly' not in df.columns:
                print("Anomaly column not found. Generating anomalies based on price changes.")
                df['price_change'] = df['price'].diff().abs()
                threshold = df['price_change'].mean() + 2.5 * df['price_change'].std()
                df['anomaly'] = (df['price_change'] > threshold).astype(int)
            
            print(f"Loaded data with {len(df)} entries")
        
        # Preprocess the data for time series
        feature_columns = ['price']
        
        # Add volume if available
        if 'volume' in df.columns:
            feature_columns.append('volume')
        
        # Get the dates for the test set
        df = df.sort_values('date')
        
        # Preprocess data
        X, y, scaler = preprocess_data(df, feature_columns=feature_columns, target_column='anomaly')
        
        # Configure sequence length for time series
        sequence_length = 30
        X_sequences, y_sequences = split_data(X, y, sequence_length)
        
        # Split into train/test sets
        train_size = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
        y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]
        
        # Get corresponding dates for test set (for visualization)
        test_start_idx = train_size + sequence_length - 1
        date_test = df['date'].iloc[test_start_idx:test_start_idx + len(y_test)].reset_index(drop=True)
        
        print(f"Prepared {len(X_train)} training sequences and {len(X_test)} test sequences")
        print(f"Sequence length: {sequence_length}, Features: {X_train.shape[2]}")
        
        return X_train, X_test, y_train, y_test, scaler, date_test
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def test_model(model_name, X_train, X_test, y_train, scaler):
    """Train and test a specific model"""
    # Set model parameters based on data shape
    input_shape = X_train.shape[1:]
    n_features = X_train.shape[2]
    
    # Initialize appropriate model
    if model_name == 'lstm':
        model = LSTMModel(input_shape)
    elif model_name == 'autoencoder':
        model = AutoEncoder(input_shape)
    elif model_name == 'hybrid':
        model = HybridCNNLSTM(input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train the model
    print(f"Training {model_name} model...")
    model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Make predictions
    print(f"Generating predictions with {model_name} model...")
    y_pred = model.predict(X_test)
    
    return y_pred, model

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics"""
    # For LSTM and Hybrid models, prediction might be probabilities
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        # For autoencoder, prediction might be reconstruction error
        # We'll use a threshold on the flattened values
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        # Convert continuous predictions to binary
        threshold = np.mean(y_pred) + np.std(y_pred)
        y_pred_classes = (y_pred > threshold).astype(int)
    
    # Ensure y_true is flattened
    y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
    
    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average='binary'
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # False positive rate and false negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    return metrics

def compare_models(results, predictions, y_test, date_test):
    """Compare model performance and visualize results"""
    print("\n2. Model Comparison")
    print("-" * 60)
    
    # Convert results to DataFrame for easier comparison
    df_metrics = pd.DataFrame(results).T
    print("Model Performance Comparison:")
    print(df_metrics[['accuracy', 'precision', 'recall', 'f1_score']].round(4))
    
    # Plot performance metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    plt.figure(figsize=(12, 8))
    df_metrics[metrics_to_plot].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../plots/model_performance_comparison.png")
    print("Saved model performance comparison to ../plots/model_performance_comparison.png")
    plt.close()
    
    # Plot confusion matrix metrics
    confusion_metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    
    plt.figure(figsize=(12, 8))
    df_metrics[confusion_metrics].plot(kind='bar', figsize=(10, 6))
    plt.title('Confusion Matrix Components by Model')
    plt.ylabel('Count')
    plt.xlabel('Model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../plots/confusion_matrix_comparison.png")
    print("Saved confusion matrix comparison to ../plots/confusion_matrix_comparison.png")
    plt.close()
    
    # Plot ROC curves (placeholder, would need prediction probabilities)
    plt.figure(figsize=(10, 8))
    plt.plot([0, 0.2, 0.5, 0.8, 1], [0, 0.6, 0.8, 0.9, 1], 'b-', label='LSTM')
    plt.plot([0, 0.3, 0.6, 0.8, 1], [0, 0.5, 0.75, 0.9, 1], 'r-', label='Autoencoder')
    plt.plot([0, 0.15, 0.4, 0.7, 1], [0, 0.65, 0.85, 0.95, 1], 'g-', label='Hybrid CNN-LSTM')
    plt.title('ROC Curves (Placeholder)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("../plots/roc_curves.png")
    print("Saved ROC curve placeholder to ../plots/roc_curves.png")
    plt.close()
    
    # Create a summary of the results
    summary = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1_Score': [],
        'Anomalies_Detected': []
    }
    
    for model_name, metrics in results.items():
        summary['Model'].append(model_name)
        summary['Accuracy'].append(metrics['accuracy'])
        summary['Precision'].append(metrics['precision'])
        summary['Recall'].append(metrics['recall'])
        summary['F1_Score'].append(metrics['f1_score'])
        summary['Anomalies_Detected'].append(metrics['true_positives'])
    
    summary_df = pd.DataFrame(summary)
    print("\nPerformance Summary:")
    print(summary_df.round(4))
    
    # Save summary to CSV
    summary_df.to_csv("../plots/model_performance_summary.csv", index=False)
    print("Saved performance summary to ../plots/model_performance_summary.csv")

def save_results(results):
    """Save detailed model results to CSV"""
    # Prepare results for saving
    all_results = []
    
    for model_name, metrics in results.items():
        row = {'model': model_name}
        row.update(metrics)
        all_results.append(row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("../plots/detailed_model_results.csv", index=False)
    
    print("Saved detailed model results to ../plots/detailed_model_results.csv")

if __name__ == "__main__":
    main() 