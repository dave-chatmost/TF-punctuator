import os

import tensorflow as tf


def inputs(data_dir, num_steps=20, batch_size=1, tfrecords_format="tfrecords-*"):
    """Construct input and label for punctuation prediction.

    Args:
        data_dir:
        num_steps:
        batch_size:

    Returns:
        input_batch: tensor of [batch_size, num_steps] 
        label_batch: tensor of [batch_size, num_steps]
    """
    MATCH_FORMAT = os.path.join(data_dir, tfrecords_format)
    files = tf.train.match_filenames_once(MATCH_FORMAT)

    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    EXAMPLES_PER_FILE = 500000
    features = tf.parse_single_example(
        serialized_example,
        features={
            "inputs": tf.FixedLenFeature([EXAMPLES_PER_FILE], tf.int64),
            "labels": tf.FixedLenFeature([EXAMPLES_PER_FILE], tf.int64),
        })

    input = features["inputs"]
    label = features["labels"]

    data_len = tf.size(input)
    batch_len = tf.to_int32(data_len / batch_size)

    input_data = tf.transpose(
                    tf.reshape(
                        tf.slice(input, [0], [batch_size * batch_len]), 
                        [batch_size, batch_len]))
    label_data = tf.transpose(
                    tf.reshape(
                        tf.slice(label, [0], [batch_size * batch_len]), 
                        [batch_size, batch_len]))

    num_threads = 16
    capacity = 10000 + 20 * batch_size

    input_batch, label_batch = tf.train.batch(
        [input_data, label_data], batch_size=num_steps, num_threads=num_threads, 
        capacity=capacity, enqueue_many=True, shapes=[[batch_size], [batch_size]])

    input_batch = tf.transpose(input_batch)
    label_batch = tf.transpose(label_batch)

    return input_batch, label_batch, files
