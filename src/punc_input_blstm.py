import os

import numpy as np
import tensorflow as tf


def inputs(data_dir, num_steps=20, batch_size=1, tfrecords_format="tfrecords-*", mode="words", fileshuf=True):
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

    filename_queue = tf.train.string_input_producer(files, shuffle=fileshuf)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    if mode == "words":
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

        num_threads = 32
        capacity = 20000 + 20 * batch_size

        input_batch, label_batch = tf.train.batch(
            [input_data, label_data], batch_size=num_steps, num_threads=num_threads, 
            capacity=capacity, enqueue_many=True, shapes=[[batch_size], [batch_size]])

        input_batch = tf.transpose(input_batch)
        label_batch = tf.transpose(label_batch)
        return input_batch, label_batch, files
    elif mode == "sentences":
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        seq_length, sequence = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        length = seq_length["length"]
        inputs = sequence["inputs"]
        labels = sequence["labels"]

        num_threads = 16
        capacity = 10000 + 20 * batch_size
        batch = tf.train.batch(
            tensors=[inputs, labels, length],
            batch_size=batch_size,
            dynamic_pad=True,
            num_threads=num_threads,
            capacity=capacity
        )
        input_batch = batch[0]
        label_batch = batch[1]
        seq_len = batch[2]
        return input_batch, label_batch, seq_len, files


def eval_inputs(inputs, outputs, lens):
    N = len(inputs)
    max_len = max(lens)
    inputs_pad = np.zeros((N, max_len))
    labels_pad = np.zeros((N, max_len))
    for i, (input, output) in enumerate(zip(inputs, outputs)):
        end = lens[i]
        inputs_pad[i, :end] = input[:end]
        labels_pad[i, :end] = output[:end]

    inputs = tf.convert_to_tensor(inputs_pad, name="inputs", dtype=tf.int32)
    labels = tf.convert_to_tensor(labels_pad, name="labels", dtype=tf.int32)
    lens = tf.convert_to_tensor(lens, name="lens", dtype=tf.int32)

    epoch_size = N

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    input_batch = tf.reshape(inputs[i][:lens[i]], [1, -1])
    label_batch = tf.reshape(labels[i][:lens[i]], [1, -1])
    seq_len = tf.reshape(lens[i], [-1])
    return input_batch, label_batch, seq_len

def get_epoch_size(pickle_file, batch_size, num_steps, EXAMPLES_PER_FILE=500000):
    data=np.load(pickle_file)
    data_len=len(data["inputs"])
    #epoch_size = (data_len // EXAMPLES_PER_FILE)*EXAMPLES_PER_FILE // batch_size // num_steps
    epoch_size = data_len // batch_size
    return epoch_size
