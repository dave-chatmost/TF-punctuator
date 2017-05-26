import os
import tensorflow as tf 

def inputs(data_dir, batch_size=1, shuffle=False, tfrecords_format="tfrecords-*"):
    """Construct input and label for punctuation prediction.

    Args:
        data_dir:
        batch_size:
        shuffle:

    Returns:
        input_batch: tensor of [batch_size, num_steps] 
        label_batch: tensor of [batch_size, num_steps]
    """
    MATCH_FORMAT = os.path.join(data_dir, tfrecords_format)
    files = tf.train.match_filenames_once(MATCH_FORMAT)

    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 20 is BPTT steps in LSTM, I just set it fixed number now.
    features = tf.parse_single_example(
        serialized_example,
        features={
            "inputs": tf.FixedLenFeature([20], tf.int64),
            "labels": tf.FixedLenFeature([20], tf.int64),
        })

    input = tf.cast(features["inputs"], tf.int32)
    label = tf.cast(features["labels"], tf.int32)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 20 * batch_size

    if shuffle:
        input_batch, label_batch = tf.train.shuffle_batch(
            [input, label], batch_size=batch_size, num_threads=8, 
            capacity=capacity, min_after_dequeue=min_after_dequeue)
    else:
        input_batch, label_batch = tf.train.batch(
            [input, label], batch_size=batch_size, num_threads=8, 
            capacity=capacity)

    return input_batch, label_batch, files
