from scipy.stats import zscore
import matplotlib.pyplot as plt

def detect_anomalies_zscore(df):
    """Detects anomalies using Z-score analysis."""
    df['z_score'] = zscore(df['close'])
    df['anomaly'] = df['z_score'].apply(lambda x: 1 if abs(x) > 3 else 0)
    return df

def plot_zscore_anomalies(df):
    """Plots anomalies detected using Z-score method."""
    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['close'], label='Price')
    plt.scatter(df[df['anomaly']==1]['timestamp'], df[df['anomaly']==1]['close'], color='red', label='Anomaly')
    plt.legend()
    plt.title("Anomaly Detection with Z-score")
    plt.show()
