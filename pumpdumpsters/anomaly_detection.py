from scipy.stats import zscore
import matplotlib.pyplot as plt

def detect_anomalies_zscore(df):
    """Detects anomalies using Z-score analysis."""
    # Use 'Close' column since the CSV now has capitalized column names
    df['z_score'] = zscore(df['Close'])
    df['anomaly'] = df['z_score'].apply(lambda x: 1 if abs(x) > 3 else 0)
    return df

def plot_zscore_anomalies(df):
    """Plots anomalies detected using Z-score method."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Price')
    plt.scatter(df[df['anomaly'] == 1]['Date'], df[df['anomaly'] == 1]['Close'], color='red', label='Anomaly')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Anomaly Detection with Z-score")
    plt.legend()
    plt.savefig("plots/zscore_anomalies.png")
    plt.show()