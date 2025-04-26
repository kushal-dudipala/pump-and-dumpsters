import tensorflow as tf
import numpy as np

def batching(X, y, batch_size):
    """
    Simple batching function using tf.data.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    batch_size : int
        Size of each batch.

    Returns:
    --------
    tf.data.Dataset
        A batched TensorFlow dataset.
    """
    assert len(X) == len(y), "Features and labels must have the same length."
    assert batch_size > 0, "Batch size must be greater than 0."

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

