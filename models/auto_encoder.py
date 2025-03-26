import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from models.utils import convert_to_numpy_format, train_test_split, batching

def train_autoencoder(
    df: pd.DataFrame,
    features: list = ['close', 'volume'],
    test_size: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer: str = 'adam',
    dropout_rate: float = 0.2
):
    """
    Trains an autoencoder model
    
    Parameters:
    -----------
    df              : DataFrame with at least a 'close' column (already normalized if needed).
    features        : List of feature columns to be used for training the autoencoder.
    test_size       : Fraction of data used for testing.
    epochs          : Number of training epochs.
    batch_size      : Training batch size.
    learning_rate   : Learning rate for the optimizer.
    optimizer       : Defaults to 'adam'. If set to 'muon', uses the custom Muon optimizer.
    dropout_rate    : Dropout rate for LSTM layers to prevent overfitting.
    
    Returns:
    --------
    model           : trained LSTM model 
    X_test          : feature sequences 
    
    Notes:
    ------
    - The model is built with a simple feedforward architecture.
    - The input shape is determined by the number of features in the data.
    - The model utilizes a mean squared error loss function.
    - The data is split into training and testing sets without shuffling to maintain time-series order.
    - The function assumes the DataFrame has the specified feature columns.
    - The function uses the specified feature columns for training the autoencoder.
    """
    for col in features:
        assert col in df.columns, f"DataFrame must contain '{col}' column."
    
    # Validate parameters
    assert 0 < test_size < 1, ("test_size must be between 0 and 1.")
    assert batch_size > 0, ("batch_size must be greater than 0.")
    assert learning_rate > 0, ("learning_rate must be greater than 0.")
    
    data = df[features].values
    data = convert_to_numpy_format(data)
    assert data.ndim == 2, ("Converted data must be a 2D array.")
    
    X_train, X_test = train_test_split(data, test_size=test_size, shuffle=False)
    assert len(X_train) > 0 and len(X_test) > 0, ("Training and test splits must be non-empty.")

    input_dim = X_train.shape[1]

    # autoencoder model architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')
    ])

    valid_optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        # add more if needed
    }
    assert optimizer in valid_optimizers, (
        f"Unknown optimizer '{optimizer}'. Must be one of: {list(valid_optimizers.keys())}"
    )
    opt_class = valid_optimizers[optimizer]
    opt = opt_class(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.MeanSquaredError()
    
    train_batches = batching(X_train, X_train, batch_size)
    test_batches  = batching(X_test, X_test, batch_size)
    
    
    for epoch in range(epochs):
        loss_per_epoch = 0.0
        for X_batch, _ in train_batches:
            with tf.GradientTape() as tape:
                preds = model(X_batch, training=True)
                loss = loss_fn(X_batch, preds)
                
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            loss_per_epoch += float(loss)
            
        loss_per_epoch /= len(train_batches)
        
        val_loss = 0.0
        for X_batch, _ in test_batches:
            preds = model(X_batch, training=False)
            loss = loss_fn(X_batch, preds)
            val_loss += float(loss)
            
        val_loss /= len(test_batches)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss_per_epoch:0.4f} | Val Loss: {val_loss:0.4f}")

    return model, X_test

def detect_autoencoder_anomalies(model, X_test, df, threshold: float = 95):
    """
    Uses the trained Autoencoder to detect anomalies.
    
    Parameters:
    -----------
    model           : Trained autoencoder model.
    X_test          : np.ndarray of test data used for anomaly detection.
    df              : Original dataframe with 'close' column.
    threshold       : Error percentile above which data points are flagged as anomalies.

    Returns:
    --------
    df              : The same dataframe but with an 'lstm_anomaly' column
    
    """
    assert 'close' in df.columns, ("DataFrame must contain 'close' column.")
    assert len(X_test) > 0, ("X_test must be non-empty.")
    
    X_pred = model.predict(X_test)
    assert X_pred.shape[0] == X_test.shape[0], ("The number of predictions must match the number of test samples.")
    mse = np.mean(np.abs(X_pred - X_test), axis=1)
    
    computed_threshold = np.percentile(mse, threshold)
    anomalies = mse > computed_threshold

    # Assume that X_test corresponds to the last N rows in df:
    n_test = len(X_test)
    df.loc[df.index[-n_test:], 'autoencoder_anomaly'] = anomalies
    df['autoencoder_anomaly'].fillna(False, inplace=True)
    
    return df