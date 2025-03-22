from sklearn.metrics import classification_report

def evaluate_model(df):
    """Compares anomalies from different methods."""
    # Convert anomalies to 0/1
    if 'autoencoder_anomaly' in df.columns:
        df['autoencoder_anomaly'] = df['autoencoder_anomaly'].fillna(False).astype(bool).astype(int)
    if 'anomaly' in df.columns:
        df['anomaly'] = df['anomaly'].fillna(False).astype(bool).astype(int)

    print(classification_report(df['anomaly'], df['autoencoder_anomaly']))