import matplotlib.pyplot as plt

def apply_mean_reversion_strategy(df):
    """Identifies trading opportunities using mean reversion strategy."""
    # Compute 30-day SMA
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # Check that both anomalies exist, or handle gracefully if one is missing
    if 'autoencoder_anomaly' not in df.columns:
        df['autoencoder_anomaly'] = False
    if 'lstm_anomaly' not in df.columns:
        df['lstm_anomaly'] = False

    # Example: Mean reversion signal if close < SMA_30 AND (autoencoder_anomaly OR lstm_anomaly)
    df['mean_reversion_signal'] = (
        (df['close'] < df['SMA_30']) &
        (df['autoencoder_anomaly'] | df['lstm_anomaly'])
    )

    return df

def plot_mean_reversion(df):
    """Plots trading signals based on mean reversion strategy."""
    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['close'], label='Price')
    plt.plot(df['timestamp'], df['SMA_30'], label='30-day SMA', color='orange')
    plt.scatter(df[df['mean_reversion_signal']==1]['timestamp'], df[df['mean_reversion_signal']==1]['close'], color='green', label='Buy Signal')
    plt.legend()
    plt.title("Mean Reversion Trading Strategy")
    plt.show()
