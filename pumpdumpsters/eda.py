import os

import matplotlib.pyplot as plt

def plot_price_trends(df, dir):
    """Plots cryptocurrency closing prices over time."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'])
    plt.title("Cryptocurrency Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.savefig(os.path.join(dir, "price_trends.png"))
    plt.show()

def plot_volume_trends(df, dir):
    """Plots trading volume over time."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Volume'], color='red')
    plt.title("Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.savefig(os.path.join(dir, "volume_trends.png"))
    plt.show()