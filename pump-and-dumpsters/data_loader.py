import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filename="cryptocurrencypricehistory.csv"):
    """Loads cryptocurrency dataset, handles missing values, and normalizes features."""
    df = pd.read_csv(filename)

    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

    return df
