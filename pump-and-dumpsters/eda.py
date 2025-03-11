import matplotlib.pyplot as plt

def plot_price_trends(df):
    """Plots cryptocurrency closing prices over time."""
    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['close'])
    plt.title("Cryptocurrency Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.show()

def plot_volume_trends(df):
    """Plots trading volume over time."""
    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['volume'], color='red')
    plt.title("Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.show()
