import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from typing import Tuple

"""
Notes: May need to create optimizer and test_train_split ourselves
"""



import numpy as np
from typing import Tuple, Union

def create_sequences(
    data:       Union[np.ndarray, list],
    seq_len:    int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 1D array of numeric values into a 2D array of sequences (X) 
    and their next-step targets (y).

    Parameters
    ----------
    data            : A sequence of numeric values (closing prices).
    seq_len         : The length of each subsequence used as input to the LSTM.

    Returns
    -------
    X               : 3D array of shape (num_samples, seq_len, 1).
    y               : 1D array of corresponding next-step targets.
    """
    # Convert data to np.float32 for consistency
    data_array = np.array(data, dtype=np.float32)

    X, y = [], []
    for i in range(len(data_array) - seq_len):
        seq = data_array[i : i + seq_len]
        target = data_array[i + seq_len]
        X.append(seq)
        y.append(target)

    # Convert lists to np.ndarray
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Reshape X to (num_samples, timesteps, features). Here features=1.
    X = np.expand_dims(X, axis=-1)

    return X, y


def train_lstm_model(
    df:         pd.DataFrame,
    seq_len:    int = 30,
    test_size:  float = 0.2,
    epochs:     int = 10,
    batch_size: int = 16
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """
    Trains a simple LSTM model 
    
    Parameters:
    -----------
    df              : DataFrame with at least a 'close' column (already normalized if needed).
    seq_len         : Length of time-series sequences fed into LSTM.
    test_size       : Fraction of data used for testing.
    epochs          : Number of training epochs.
    batch_size      : Training batch size.

    Returns:
    --------
    model           :  trained LSTM model 
    X_test          : feature sequences 
    y_test          : test targets 
    """
    close_prices = df['close'].values
    X, y = create_sequences(close_prices, seq_len=seq_len)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Build LSTM model
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64),
        Dense(1)
    ])

    # Add optimizer (ADAM) and loss function (MSE)
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, X_test, y_test

def detect_lstm_anomalies(model, df, seq_len=30, percentile_threshold=95):
    """
    Uses the trained LSTM model to detect anomalies by comparing predicted 
    vs. actual closing prices. Any prediction error above the chosen threshold 
    is flagged as an anomaly.
    
    Parameters:
    -----------
    model                   : Trained LSTM model.
    df                      : Original dataframe with 'close' column.
    seq_len                 : Must match the sequence length used during training.
    percentile_threshold    : Error percentile above which data points are flagged as anomalies.

    Returns:
    --------
    df                      : The same dataframe but with an 'lstm_anomaly' column 
    """
    close_prices = df['close'].values
    X_all, y_all = create_sequences(close_prices, seq_len=seq_len)
    
    # Predict
    predictions = model.predict(X_all)
    errors = np.abs(predictions.flatten() - y_all)
    
    # Compute threshold based on the specified percentile of errors
    threshold = np.percentile(errors, percentile_threshold)
    
    # Anomaly flags: For the first `seq_len` rows, we can't have predictions
    anomalies = [False]*seq_len + [err > threshold for err in errors]
    
    df['lstm_anomaly'] = anomalies
    return df
