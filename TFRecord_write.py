import collections
import os
import numpy as np
import tensorflow as tf
import pickle

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data",
                    """Directory where exists text dataset""")
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
    return inputs, outputs


def convert_file(file_path, vocabulary, punctuations, output_path):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    print("Converting " + file_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    inputs, outputs = words_to_ids(file_path, vocabulary, punctuations)
    print("Length of inputs is " + str(len(inputs)))
    assert len(inputs) == len(outputs)

    data = {"inputs": inputs, "outputs": outputs,
            "vocabulary": vocabulary, "punctuations": punctuations}
    
    with open(output_path+".pkl", 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    EXAMPLES_PER_FILE = 20000000 #2000W
    NUM_FILES = int(np.ceil(len(inputs)/EXAMPLES_PER_FILE))
    EXAMPLES_PER_FEATURE = 20 # num_steps
    NUM_FEATURES = EXAMPLES_PER_FILE // EXAMPLES_PER_FEATURE # Make sure divisible
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        print("Writing " + filename)
        for j in range(NUM_FEATURES):
            if i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE > len(inputs):
                break
            input = inputs[i*EXAMPLES_PER_FILE + j*EXAMPLES_PER_FEATURE : i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE]
            label = outputs[i*EXAMPLES_PER_FILE + j*EXAMPLES_PER_FEATURE : i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE]
            example = tf.train.Example(features=tf.train.Features(feature={
                "inputs": _int64_feature(input),
                "labels": _int64_feature(label)}))
            writer.write(example.SerializeToString())
        writer.close()
    print("Converting Successfully.")


def convert_text_to_tfrecord(raw_data_path, conf, output_dir="data"):
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
    convert_file(train_data, vocabulary, punctuations,
                os.path.join(data_path, "train"))
    convert_file(valid_data, vocabulary, punctuations,
                os.path.join(data_path, "valid"))
    convert_file(test_data, vocabulary, punctuations,
                os.path.join(data_path, "test"))


def main():
    convert_text_to_tfrecord(FLAGS.data_dir, conf=Conf())


if __name__ == "__main__":
    main()
