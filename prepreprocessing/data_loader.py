import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from prepreprocessing.feature_learning import extract_features

def load_and_preprocess_data(filename=None):
    """
    Loads cryptocurrency dataset, handles missing values, and normalizes features.
    
    Parameters:
    -----------
    filename : Path to the CSV file to load. If None, defaults to the sample data.
    
    Returns:
    --------
    df : DataFrame with preprocessed cryptocurrency data
    """
    # If no filename provided, use the sample data
    if filename is None or filename == "test.csv":
        # Look for sample data in the data/raw directory
        default_paths = [
            "data/raw/coin_Bitcoin.csv",  # From project root
            "../data/raw/coin_Bitcoin.csv",  # If running from inside pumpdumpsters
            "coin_Bitcoin.csv"  # Directly in current directory
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                filename = path
                print(f"Using sample data: {path}")
                break
        else:
            raise FileNotFoundError(
                "Could not find sample data. Please either provide a valid file path "
                "or ensure sample_bitcoin.csv exists in data/raw directory."
            )
    
    # Load the data
    print(f"Loading data from: {filename}")
    df = pd.read_csv(filename)
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Handle missing values with forward fill
    df.ffill(inplace=True)

    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset. "
                            f"Available columns: {df.columns.tolist()}")
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Close', 'Volume']]
    )
    
    df = extract_features(df)
    
    # Optionally, sort the DataFrame by date to ensure time-series order
    df.sort_values(by=['Symbol', 'Date'], inplace=True)
    
    print(f"Loaded data: {len(df)} rows, columns: {df.columns.tolist()}")
    print(f"Symbols: {df['Symbol'].unique().tolist()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df
