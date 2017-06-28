import os

import numpy as np
import tensorflow as tf


def inputs(data_dir, num_steps=20, batch_size=1, tfrecords_format="tfrecords-*", mode="words"):
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

    filename_queue = tf.train.string_input_producer(files, shuffle=True)

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

        num_threads = 32
        capacity = 20000 + 20 * batch_size
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
        #i = tf.constant(0, dtype=tf.int32)

        #i = tf.train.range_input_producer(tf.size(inputs_batch[0])//num_steps, shuffle=False).dequeue()
        #input_batch = tf.strided_slice(inputs_batch, [0, i * num_steps],
        #                               [batch_size, (i + 1) * num_steps])
        #label_batch = tf.strided_slice(labels_batch, [0, i * num_steps],
        #                               [batch_size, (i + 1) * num_steps])
        return input_batch, label_batch, seq_len, files



def eval_inputs(data_dir, batch_size=1, inputs=None, outputs=None):
    """Construct input and label for punctuation evaluation.

    Args:
        data_dir: the path of pickle file. 
        batch_size:
        inputs/outputs: Used by punctuate_text_with_lstm.py, raw word id sequence.

    Returns:
        input_batch:
        label_batch:
    """
    if inputs == None:
        eval = np.load(data_dir)
        eval_inputs = eval["inputs"]
        eval_labels = eval["outputs"]
    else:
        eval_inputs = inputs
        eval_labels = outputs

    eval_inputs = tf.convert_to_tensor(eval_inputs, name="eval_inputs", dtype=tf.int32)
    eval_labels = tf.convert_to_tensor(eval_labels, name="eval_labels", dtype=tf.int32)

    data_len = tf.size(eval_inputs)
    batch_len = data_len // batch_size
    inputs_data = tf.reshape(eval_inputs[0: batch_size * batch_len],
                             [batch_size, batch_len])
    labels_data = tf.reshape(eval_labels[0: batch_size * batch_len],
                             [batch_size, batch_len])

    num_steps = 1 # Fixed when evaluate
    epoch_size = batch_len // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    input_batch = tf.strided_slice(inputs_data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    input_batch.set_shape([batch_size, num_steps])
    label_batch = tf.strided_slice(labels_data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    label_batch.set_shape([batch_size, num_steps])
    return input_batch, label_batch


def get_epoch_size(pickle_file, batch_size, num_steps, EXAMPLES_PER_FILE=500000):
    data=np.load(pickle_file)
    data_len=len(data["inputs"])
    epoch_size = (data_len // EXAMPLES_PER_FILE)*EXAMPLES_PER_FILE // batch_size // num_steps
    return epoch_size
