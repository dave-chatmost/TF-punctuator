# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import pickle

import tensorflow as tf


# ******* MY CODE START *******
def input_word_index(vocabulary, input_word):
    return vocabulary.get(input_word, vocabulary["<unk>"])

def punctuation_index(punctuations, punctuation):
    return punctuations[punctuation]

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()
    #return f.read().decode("utf-8").replace("\n", "<eos>").split()

def build_vocab(corpus, vocab_size, output_file):
  data = _read_words(corpus)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words = list(zip(*count_pairs))[0][:vocab_size]
  # word_to_id = dict(zip(words, range(len(words))))
  with open(output_file, "w") as vocab:
    for i in range(len(words)):
      vocab.write(words[i] + "\n")


def load_vocabulary(file_path):
  with open(file_path, 'r') as vocab:
    vocabulary = {w.strip(): i for (i, w) in enumerate(vocab)}
  if "<unk>" not in vocabulary:
    vocabulary["<unk>"] = len(vocabulary)
  if "<END>" not in vocabulary:
    vocabulary["<END>"] = len(vocabulary)
  return vocabulary


def convert_file(file_path, vocabulary, punctuations, output_path):
  inputs = []
  outputs = []
  punctuation = " "
  print("[DEBUG]", punctuations)
  
  with open(file_path, 'r') as corpus:
    for line in corpus:
      for token in line.split():
        if token in punctuations:
          punctuation = token
          continue
        else:
          inputs.append(input_word_index(vocabulary, token))
          outputs.append(punctuation_index(punctuations, punctuation))
          punctuation = " "

  inputs.append(input_word_index(vocabulary, "<END>"))
  outputs.append(punctuation_index(punctuations, punctuation))

  assert(len(inputs) == len(outputs))

  data = {"inputs": inputs, "outputs": outputs,
          "vocabulary": vocabulary, "punctuations": punctuations}
  
  with open(output_path, 'wb') as output_file:
    pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def convert_data(raw_data_path, conf):
  vocab_file = os.path.join(raw_data_path, conf.vocabulary_file)
  train_data = os.path.join(raw_data_path, conf.train_data)
  valid_data = os.path.join(raw_data_path, conf.valid_data)
  test_data = os.path.join(raw_data_path, conf.test_data)
  data_path = os.path.join(raw_data_path, "data")

  vocabulary = load_vocabulary(vocab_file)
  convert_file(train_data, vocabulary, conf.punctuations,
               os.path.join(data_path, "train"))
  convert_file(valid_data, vocabulary, conf.punctuations,
               os.path.join(data_path, "valid"))
  convert_file(test_data, vocabulary, conf.punctuations,
               os.path.join(data_path, "test"))


def load_data(raw_data_path, conf):
  vocab_file = os.path.join(raw_data_path, conf.vocabulary_file)
  if not os.path.exists(vocab_file):
    print("Vocabulary 'vocab' does not exist.\n")
    print("Building vocab from train data (train.txt)")
    train_data = os.path.join(raw_data_path, conf.train_data)
    build_vocab(train_data, conf.vocab_size, vocab_file)
    print("Build vocab successfully.\n")

  data_path = os.path.join(raw_data_path, "data")
  if not os.path.exists(data_path):
    print("Converted data directory 'data' does not exist.\n")
    print("Converting data...\n")
    os.makedirs(data_path)
    convert_data(raw_data_path, conf)
    print("Convert data successfully.\n")
  # return 0,0,0 # DEBUG

  print("Loading data...\n")
  # data stored as a dict:
  # data = {"inputs": inputs, "outputs": outputs,
  #         "vocabulary": vocabulary, "punctuations": punctuations}
  train_data = np.load(os.path.join(data_path, "train.pkl"))
  valid_data = np.load(os.path.join(data_path, "valid.pkl"))
  test_data = np.load(os.path.join(data_path, "test.pkl"))
  print("Loading data successfully.\n")
  return train_data, valid_data, test_data


def batch_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw punctuation data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from load_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. 
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "BatchProducer", [raw_data, batch_size, num_steps]):
    inputs = raw_data["inputs"]
    outputs = raw_data["outputs"]
    assert(len(inputs) == len(outputs))
    # print("[DEBUG]", len(inputs))
    # print("[DEBUG]", len(outputs))
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    data_len = inputs.size
    batch_len = data_len // batch_size
    epoch_size = batch_len // num_steps

    inputs_data = inputs[0 : batch_size * batch_len].reshape(batch_size, batch_len).T
    outputs_data = outputs[0 : batch_size * batch_len].reshape(batch_size, batch_len).T

    queue = tf.FIFOQueue(capacity=16000, dtypes=[tf.int32, tf.int32], shapes=[[batch_size],[batch_size]])

    enqueue_op = queue.enqueue_many([inputs_data, outputs_data])
    data_sample, label_sample = queue.dequeue_many(num_steps)
    data_sample = tf.transpose(data_sample)
    label_sample = tf.transpose(label_sample)

    NUM_THREADS = 8
    qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
    return data_sample, label_sample, qr

"""
    inputs = tf.convert_to_tensor(inputs, name="inputs", dtype=tf.int32)
    outputs = tf.convert_to_tensor(outputs, name="outputs", dtype=tf.int32)

    data_len = tf.size(inputs)
    batch_len = data_len // batch_size
    inputs_data = tf.reshape(inputs[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    outputs_data = tf.reshape(outputs[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = batch_len // num_steps # don't need to minus 1
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(inputs_data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(outputs_data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    y.set_shape([batch_size, num_steps])
    return x, y
"""

# ******* MY CODE END *******