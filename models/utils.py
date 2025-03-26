import numpy as np
from typing import Tuple, Optional, Union


def convert_to_numpy_format(floats: list) -> np.ndarray:
    """
    Converts a list of floats to a numpy array of type float32.
    
    """
    return np.array(floats, dtype=np.float32)

def train_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    shuffle: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Splits arrays X and y into train and test subsets.
 
    Parameters
    ----------
    X               : Feature array, of shape (n_samples, ...)
    y               : Target array, of shape (n_samples, ...)
    test_size       : Fraction of the data to reserve for test, default 20%
    shuffle         : Whether to shuffle the data before splitting, default False.
 
    Returns
    -------
    If y is provided:
        X_train     : np.ndarray
        X_test      : np.ndarray
        y_train     : np.ndarray
        y_test      : np.ndarray
    If y is not provided:
        X_train     : np.ndarray
        X_test      : np.ndarray
    """
    n_samples = len(X)
    test_count = int(n_samples * test_size)
    
    if shuffle:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        if y is not None:
            y = y[indices]
    
    # The training set is the first (n_samples - test_count) samples,
    # and the test set is the remaining test_count samples.
    split_index = n_samples - test_count
    X_train = X[:split_index]
    X_test = X[split_index:]
    
    if y is not None:
        y_train = y[:split_index]
        y_test = y[split_index:]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test

def batching(X: np.ndarray, y: np.ndarray, batch_size: int):
    """
    Splits the arrays X and y into mini-batches.

    Parameters
    ----------
    X : The feature array of shape (n_samples, ...).
    y : The target array of shape (n_samples, ...).
    batch_size : The size of each batch.

    Returns
    -------
    batches : A list where each tuple is (X_batch, y_batch)
    """
    assert len(X) == len(y), ("X and y must have the same number of samples.")
    batches = []
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        X_batch = X[i: i + batch_size]
        y_batch = y[i: i + batch_size]
        batches.append((X_batch, y_batch))
    return batches