import argparse
import traceback
import os
from models.lstm_model import train_lstm_model, detect_lstm_anomalies
from models.auto_encoder import train_autoencoder, detect_autoencoder_anomalies
from models.hybrid_cnn_lstm import train_hybrid_cnn_lstm_model, detect_hybrid_anomalies
from prepreprocessing.data_loader import load_and_preprocess_data

parser = argparse.ArgumentParser(description="Train a model and detect anomalies.")
parser.add_argument(
    "--model",
    required=True,
    choices=["lstm", "cnn", "hybrid"],
    help="Which model to train: 'lstm', 'cnn', or 'hybrid'."
)
parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for LSTM/Hybrid (default=30)")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default=20)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default=16)")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default=0.001)")
parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default=0.2)")
parser.add_argument("--cnn_filters", type=int, default=64, help="CNN filters for Hybrid model (default=64)")
parser.add_argument("--cnn_kernel_size", type=int, default=3, help="CNN kernel size for Hybrid model (default=3)")
parser.add_argument("--lstm_units", type=int, default=64, help="LSTM units for LSTM/Hybrid (default=64)")
parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (default=0.2)")
parser.add_argument("--filename", type=str, default=None, help="Custom CSV file to load instead of default Bitcoin data")

args = parser.parse_args()

def train(model_type, df, **kwargs):
    """
    Trains the specified model and detects anomalies.
    """
    if model_type == "lstm":
        model, X_test, y_test = train_lstm_model(df, **kwargs)
        df_with_anomalies = detect_lstm_anomalies(model, df.copy(), seq_len=kwargs.get("seq_len", 30))
    elif model_type == "cnn":
        model, X_test = train_autoencoder(df, **kwargs)
        df_with_anomalies = detect_autoencoder_anomalies(model, X_test, df.copy())
    elif model_type == "hybrid":
        model, X_test, y_test = train_hybrid_cnn_lstm_model(df, **kwargs)
        df_with_anomalies = detect_hybrid_anomalies(model, df.copy(), seq_len=kwargs.get("seq_len", 30))
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Must be one of ['lstm', 'cnn', 'hybrid'].")

    return model, df_with_anomalies

try:
    print(f"\n=== STARTING {args.model.upper()} MODEL TRAINING ===")

    if args.filename is None:
        print("No --filename provided. Using merged processed file...")

        # Dynamically set full absolute path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        merged_file = os.path.join(project_root, "data", "processed", "merged_cryptos.csv")

        print(f"Loading data from: {merged_file}")
        df = load_and_preprocess_data(filename=merged_file)
    else:
        df = load_and_preprocess_data(filename=args.filename)
    
    # --- CONTINUES ---
    model, df_with_anomalies = train(
        args.model,
        df,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        cnn_filters=args.cnn_filters,
        cnn_kernel_size=args.cnn_kernel_size,
        lstm_units=args.lstm_units,
        test_size=args.test_size
    )

    print(f"\n=== {args.model.upper()} MODEL TRAINING COMPLETE ===")
    print("Anomalies detected and added to the DataFrame!")

except Exception as e:
    print("\n=== ERROR DURING TRAINING ===")
    print(f"Exception: {e}")
    traceback.print_exc()
