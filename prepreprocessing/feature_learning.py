import pandas as pd
import numpy as np

def extract_features(df):
    """
    Extracts simple features from Bitcoin-like CSV price data.

    Args:
        df (pd.DataFrame): DataFrame with columns ['High', 'Low', 'Open', 'Close', 'Volume'] at least.

    Returns:
        pd.DataFrame: DataFrame with new feature columns.
    """

    features = df.copy()

    # Convert 'Date' to datetime if it's not already
    if not np.issubdtype(features['Date'].dtype, np.datetime64):
        features['Date'] = pd.to_datetime(features['Date'])

    # Sort by date just in case
    features = features.sort_values('Date')

    # Price-based features
    features['return_1'] = features['Close'].pct_change()  # Daily return
    features['return_5'] = features['Close'].pct_change(periods=5)  # 5-day return

    features['high_low_diff'] = features['High'] - features['Low']  # Daily range
    features['open_close_diff'] = features['Open'] - features['Close']  # Daily net move

    features['ma_5'] = features['Close'].rolling(window=5).mean()  # 5-day moving average
    features['ma_10'] = features['Close'].rolling(window=10).mean()  # 10-day moving average

    features['volatility_5'] = features['Close'].rolling(window=5).std()  # 5-day volatility
    features['volatility_10'] = features['Close'].rolling(window=10).std()  # 10-day volatility

    # Volume features
    features['volume_change_1'] = features['Volume'].pct_change()
    features['volume_ma_5'] = features['Volume'].rolling(window=5).mean()
    features['volume_ma_10'] = features['Volume'].rolling(window=10).mean()

    # Drop rows with NaNs created by rolling functions
    features = features.dropna()

    return features

if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv('bitcoin_data.csv')
    features = extract_features(df)
    print(features.head())
