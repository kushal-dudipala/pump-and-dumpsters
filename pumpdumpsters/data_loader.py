import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filename="cryptocurrencypricehistory.csv"):
    """Loads cryptocurrency dataset, handles missing values, and normalizes features."""
    df = pd.read_csv(filename)

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Handle missing values with forward fill
    df.ffill(inplace=True)

    # Normalize numerical features (using capitalized column names)
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Close', 'Volume']]
    )
    
    # Optionally, sort the DataFrame by date to ensure time-series order
    df.sort_values(by='Date', inplace=True)

    return df
