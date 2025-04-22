# to be coded :)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout
from typing import Tuple, Union
from models.utils import convert_to_numpy_format, train_test_split, batching
from tensorflow.keras.optimizers import Adam, SGD


def create_sequences_multi(
    data: np.ndarray,
    seq_len: int = 30,
    target_col: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 2D array of numeric values (multiple features) into:
      1) a 3D array of input sequences (X)
      2) a 1D array of corresponding next-step targets (y)
    
    Parameters
    ----------
    data            : 2D array of shape (time_steps, num_features).
    seq_len         : The length of each subsequence used as input to the model.
    target_col      : Column index in `data` to use as the target.

    Returns
    -------
    X               : 3D array of shape (num_samples, seq_len, num_features).
    y               : 1D array of next-step targets from the specified feature column.
    """
    data_array = convert_to_numpy_format(data)
    assert data_array.ndim == 2, "Input data must be a 2D array."
    assert seq_len > 0, "Sequence length must be greater than 0."
    assert len(data_array) > seq_len, "Data must have more rows than the sequence length."
    
    X, Y = [], []

    for i in range(len(data_array) - seq_len):
        seq = data_array[i : i + seq_len, :]  # shape (seq_len, num_features)
        target = data_array[i + seq_len, target_col]
        X.append(seq)
        Y.append(target)

    X = convert_to_numpy_format(X)
    Y = convert_to_numpy_format(Y)

    return X, Y


def train_hybrid_cnn_lstm_model(
    df: pd.DataFrame,
    seq_len: int = 30,
    test_size: float = 0.2,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,   
    optimizer: str = 'adam',
    dropout_rate: float = 0.2,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
    pool_size: int = 2,
    lstm_units: int = 64
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """
    Trains a hybrid CNN-LSTM model for time series prediction
    
    Parameters:
    -----------
    df              : DataFrame with required columns (already normalized if needed).
    seq_len         : Length of time-series sequences fed into model.
    test_size       : Fraction of data used for testing.
    epochs          : Number of training epochs.
    batch_size      : Training batch size.
    learning_rate   : Learning rate for the optimizer.
    optimizer       : 'adam' or 'sgd'.
    dropout_rate    : Dropout rate to prevent overfitting.
    cnn_filters     : Number of filters in the Conv1D layer.
    cnn_kernel_size : Size of the kernel in the Conv1D layer.
    pool_size       : Size of the pooling window in MaxPooling1D.
    lstm_units      : Number of units in the LSTM layer.
    
    Returns:
    --------
    model           : Trained hybrid CNN-LSTM model 
    X_test          : Test feature sequences 
    y_test          : Test targets 
    
    Notes:
    ------
    - The model combines CNN layers for feature extraction with LSTM for temporal dependencies.
    - The CNN extracts local patterns while the LSTM captures long-term dependencies.
    - This hybrid approach can be more effective for complex time series data than either model alone.
    """
    assert 0 < test_size < 1, "test_size must be between 0 and 1."
    assert batch_size > 0, "batch_size must be greater than 0."
    assert learning_rate > 0, "learning_rate must be greater than 0."
    assert epochs > 0, "epochs must be greater than 0."
    assert 0 <= dropout_rate < 1, "dropout_rate must be between 0 and 1."
    
    # 1. Sort dataframe by date
    df = df.sort_values(by='Date')
    
    # 2. One-hot encode the Symbol column
    df_onehot = pd.get_dummies(df, columns=['Symbol'], prefix='Sym')
    
    # 3. Build feature list
    base_features = ['High', 'Low', 'Open', 'Close', 'Volume']
    for col in base_features:
        assert col in df.columns, f"DataFrame must contain '{col}' column."
        
    symbol_columns = [col for col in df_onehot.columns if col.startswith('Sym_')]
    feature_columns = base_features + symbol_columns
    assert 'Close' in feature_columns, "Missing 'Close' in features."
    data_features = df_onehot[feature_columns].values
    target_col = feature_columns.index('Close')

    # 4. Create sequences
    X, y = create_sequences_multi(data_features, seq_len=seq_len, target_col=target_col)

    # 5. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    assert len(X_train) > 0 and len(X_test) > 0, "Train and test splits must be non-empty."
    
    num_features = X.shape[-1]
    
    # Build hybrid model architecture
    # Input layer
    input_layer = Input(shape=(seq_len, num_features))
    
    # Set CNN parameters based on sequence length
    # For very short sequences, we need to adjust pooling to avoid dimensionality issues
    adjusted_pool_size = min(pool_size, seq_len // 2) if seq_len > 2 else 1
    
    # CNN part for spatial feature extraction with adjusted pooling
    cnn = Conv1D(filters=cnn_filters, kernel_size=min(cnn_kernel_size, seq_len-1), 
                activation='relu', padding='same')(input_layer)
    if adjusted_pool_size > 1:
        cnn = MaxPooling1D(pool_size=adjusted_pool_size)(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    
    # LSTM part for temporal dependencies
    lstm = LSTM(lstm_units, return_sequences=False, dropout=dropout_rate, 
               recurrent_dropout=dropout_rate)(cnn)
    
    # Output layer
    output = Dense(1)(lstm)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    
    # Select optimizer
    valid_optimizers = {
        'adam': Adam,
        'sgd': SGD
    }
    assert optimizer in valid_optimizers, f"Unknown optimizer '{optimizer}'. Must be one of: {list(valid_optimizers.keys())}"
    opt = valid_optimizers[optimizer](learning_rate=learning_rate)

    # Define loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Create batches
    train_dataset = batching(X_train, y_train, batch_size)
    test_dataset = batching(X_test, y_test, batch_size)
    
    assert len(train_dataset) > 0, "Training dataset batches are empty."
    assert len(test_dataset) > 0, "Test dataset batches are empty."

    # Training loop
    for epoch in range(epochs):
        loss_per_epoch = 0.0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = loss_fn(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            loss_per_epoch += float(loss)

        loss_per_epoch /= len(train_dataset)

        val_loss = 0.0
        for x_val, y_val in test_dataset:
            val_preds = model(x_val, training=False)
            val_loss += float(loss_fn(y_val, val_preds))
        val_loss /= len(test_dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss_per_epoch:.4f} | Val Loss: {val_loss:.4f}")

    return model, X_test, y_test


def detect_hybrid_anomalies(model, df, seq_len=30, threshold=95):
    """
    Uses the trained hybrid CNN-LSTM model to detect anomalies by comparing predicted 
    vs. actual closing prices. Any prediction error above the chosen threshold 
    is flagged as an anomaly.
    
    Parameters:
    -----------
    model           : Trained hybrid CNN-LSTM model.
    df              : Original dataframe with required columns.
    seq_len         : Must match the sequence length used during training.
    threshold       : Error percentile above which data points are flagged as anomalies.

    Returns:
    --------
    df              : The same dataframe but with a 'hybrid_anomaly' column 
    """
    df = df.sort_values(by='Date')

    df_onehot = pd.get_dummies(df, columns=['Symbol'], prefix='Sym')

    base_features = ['High', 'Low', 'Open', 'Close', 'Volume']
    symbol_columns = [col for col in df_onehot.columns if col.startswith('Sym_')]
    feature_columns = base_features + symbol_columns

    assert 'Close' in feature_columns, "Missing 'Close' in features."

    data_features = df_onehot[feature_columns].values
    target_col = feature_columns.index('Close')

    X_all, y_all = create_sequences_multi(data_features, seq_len=seq_len, target_col=target_col)
    assert len(X_all) > 0 and len(y_all) > 0, "Sequence creation resulted in empty arrays."

    predictions = model.predict(X_all)
    errors = np.abs(predictions.flatten() - y_all)
    assert len(predictions.flatten()) == len(y_all), "Prediction and target array lengths do not match."

    computed_threshold = np.percentile(errors, threshold)
    anomalies = [False]*seq_len + [err > computed_threshold for err in errors]

    df['hybrid_anomaly'] = anomalies
    return df