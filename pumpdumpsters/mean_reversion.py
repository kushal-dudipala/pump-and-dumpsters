import matplotlib.pyplot as plt
import os

def apply_mean_reversion_strategy(df):
    """Identifies trading opportunities using a mean reversion strategy.
    
    This function calculates a 30-day simple moving average (SMA) for the closing prices,
    and generates a trading signal if the Close is below the SMA and either an autoencoder
    or LSTM anomaly is detected.
    """
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    if 'autoencoder_anomaly' not in df.columns:
        df['autoencoder_anomaly'] = False
    if 'lstm_anomaly' not in df.columns:
        df['lstm_anomaly'] = False

    df['mean_reversion_signal'] = (
        (df['Close'] < df['SMA_30']) &
        (df['autoencoder_anomaly'] | df['lstm_anomaly'])
    )
    return df

def plot_mean_reversion(df, dir):
    """Plots trading signals based on the mean reversion strategy."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Price')
    plt.plot(df['Date'], df['SMA_30'], label='30-day SMA', color='orange')
    plt.scatter(
        df[df['mean_reversion_signal'] == 1]['Date'], 
        df[df['mean_reversion_signal'] == 1]['Close'], 
        color='green', 
        label='Buy Signal'
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Mean Reversion Trading Strategy")
    plt.legend()
    plt.savefig(os.path.join(dir, "mean_reversion_strategy.png"))
    plt.show()