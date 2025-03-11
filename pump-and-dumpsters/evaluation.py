from sklearn.metrics import classification_report

def evaluate_model(df):
    """Compares Autoencoder vs Z-score anomalies."""
    print(classification_report(df['anomaly'], df['autoencoder_anomaly']))
