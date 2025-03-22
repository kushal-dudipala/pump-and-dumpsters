import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


"""
Copied script from old code, must be modified
"""

def train_autoencoder(df):
    """Trains an Autoencoder model for anomaly detection."""
    X_train, X_test = train_test_split(df[['close', 'volume']], test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(X_train.shape[1], activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=20, batch_size=32, validation_data=(X_test, X_test))

    return model, X_test

def detect_autoencoder_anomalies(model, X_test, df):
    """Uses trained Autoencoder to detect anomalies."""
    X_pred = model.predict(X_test)
    mse = np.mean(np.abs(X_pred - X_test), axis=1)
    
    threshold = np.percentile(mse, 95)
    df['autoencoder_anomaly'] = mse > threshold
    
    return df
