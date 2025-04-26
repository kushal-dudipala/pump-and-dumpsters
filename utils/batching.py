import tensorflow as tf
import numpy as np

def batching(X, y, batch_size, num_workers=2):
    """
    Splits data into batches and distributes across workers on a single GPU.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    batch_size : int
        Size of each batch.
    num_workers : int
        Number of workers to split the data across (default: 2).

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow dataset with distributed batches.
    """
    assert len(X) == len(y), "Features and labels must have the same length."
    assert batch_size > 0, "Batch size must be greater than 0."
    assert num_workers > 0, "Number of workers must be greater than 0."

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)

    # Distribute across workers
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    # Use tf.distribute to split across workers
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    with strategy.scope():
        dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset
