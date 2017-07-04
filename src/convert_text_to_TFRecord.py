import collections
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data",
                    """Directory where exists text dataset""")
flags.DEFINE_string("mode", "words",
                    """words | sentences""")
flags.DEFINE_string("out_dir", "data",
                    """data | sentence_data""")
FLAGS = flags.FLAGS

class Conf(object):
    vocabulary_file = "vocab" # relative path to raw_data_path
    punct_vocab_file = "punct_vocab" # relative path to raw_data_path
    train_data = "train.txt" # relative path to raw_data_path
    valid_data = "valid.txt" # relative path to raw_data_path
    test_data = "test.txt" # relative path to raw_data_path
    vocab_size = 100000 # will add 2 special symbols

def get_punctuations(punct_vocab_file):
    with open(punct_vocab_file, 'r') as f:
        punctuations = {w.strip('\n'): i for (i, w) in enumerate(f)}
    return punctuations

def input_word_index(vocabulary, input_word):
    return vocabulary.get(input_word, vocabulary["<unk>"])

def punctuation_index(punctuations, punctuation):
    return punctuations[punctuation]

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(corpus, vocab_size, output_file):
    data = read_words(corpus)

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


def words_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    masks = []
    punctuation = " "
    # print("[DEBUG]", punctuations)
    
    with open(file_path, 'r') as corpus:
        for line in corpus:
            # There are some punctuations in the begin of a sentence
            meet_first_word = False
            masks.append(0)
            for token in line.split():
                if token in punctuations and meet_first_word:
                    punctuation = token
                    continue
                else:
                    meet_first_word = True
                    masks.append(1)
                    inputs.append(input_word_index(vocabulary, token))
                    outputs.append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
            inputs.append(input_word_index(vocabulary, "<END>"))
            outputs.append(punctuation_index(punctuations, punctuation))
            punctuation = " "
    return inputs, outputs, masks


def sentences_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    punctuation = " "
    
    with open(file_path, 'r') as corpus:
        for line in corpus:
            inputs.append([])
            outputs.append([])
            for token in line.split():
                if token in punctuations:
                    punctuation = token
                    continue
                else:
                    inputs[-1].append(input_word_index(vocabulary, token))
                    outputs[-1].append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
            inputs[-1].append(input_word_index(vocabulary, "<END>"))
            outputs[-1].append(punctuation_index(punctuations, punctuation))
    return inputs, outputs


def save_to_pickle(inputs, outputs, masks, vocabulary, punctuations, output_path):
    data = {"inputs": inputs, "outputs": outputs, "masks": masks,
            "vocabulary": vocabulary, "punctuations": punctuations}
    
    with open(output_path+".pkl", 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def convert_file_according_words(file_path, vocabulary, punctuations, output_path):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    print("Converting " + file_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    inputs, outputs, masks = words_to_ids(file_path, vocabulary, punctuations)
    print("Length of inputs is " + str(len(inputs)))
    assert len(inputs) == len(outputs)
    assert len(inputs) == len(masks)

    save_to_pickle(inputs, outputs, masks, vocabulary, punctuations, output_path)

    EXAMPLES_PER_FILE = 500000 #TODO: Make it an parameter
    NUM_FILES = int(np.floor(len(inputs)/EXAMPLES_PER_FILE))
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        input = inputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        label = outputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        mask = masks[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        print("Writing " + filename + " with length of " + str(len(input)) + " data.")
        example = tf.train.Example(features=tf.train.Features(feature={
            "inputs": _int64_feature(input),
            "labels": _int64_feature(label),
            "masks": _int64_feature(mask)}))
        writer.write(example.SerializeToString())
        writer.close()
    print("Converting Successfully.")


def convert_text_to_tfrecord(raw_data_path, conf, mode="words", output_dir="data"):
    vocab_file = os.path.join(raw_data_path, conf.vocabulary_file)
    punct_vocab_file = os.path.join(raw_data_path, conf.punct_vocab_file)
    train_data = os.path.join(raw_data_path, conf.train_data)
    valid_data = os.path.join(raw_data_path, conf.valid_data)
    test_data = os.path.join(raw_data_path, conf.test_data)
    data_path = os.path.join(raw_data_path, output_dir)

    if not os.path.exists(vocab_file):
        build_vocab(train_data, conf.vocab_size, vocab_file)
    vocabulary = load_vocabulary(vocab_file)
    punctuations = get_punctuations(punct_vocab_file)

    print("Converting text file according %s..." % mode)
    if mode == "words":
        convert_file = convert_file_according_words

    convert_file(train_data, vocabulary, punctuations,
                os.path.join(data_path, "train"))
    convert_file(valid_data, vocabulary, punctuations,
                os.path.join(data_path, "valid"))
    convert_file(test_data, vocabulary, punctuations,
                os.path.join(data_path, "test"))


def main():
    convert_text_to_tfrecord(FLAGS.data_dir, conf=Conf(), mode=FLAGS.mode, output_dir=FLAGS.out_dir)


if __name__ == "__main__":
    main()
