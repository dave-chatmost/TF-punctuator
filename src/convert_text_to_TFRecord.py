import collections
import os
import pickle

import numpy as np
import tensorflow as tf

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
    # print("[DEBUG]", punctuations)
    
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


def sentences_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    
    with open(file_path, 'r') as corpus:
        for line in corpus:
            # Skip blank line
            if len(line.strip()) == 0:
                continue
            punctuation = " "
            inputs.append([])
            outputs.append([])
            meet_first_word = False
            for token in line.split():
                if token in punctuations and meet_first_word:
                    punctuation = token
                    # if the length of this sentence is bigger than 100, 
                    # truncate this sentence to some short sentences.
                    if len(inputs[-1]) >= 100:
                        inputs[-1].append(input_word_index(vocabulary, "<END>"))
                        outputs[-1].append(punctuation_index(punctuations, punctuation))
                        punctuation = " "
                        inputs.append([])
                        outputs.append([])
                        meet_first_word = False
                    continue
                else:
                    meet_first_word = True
                    inputs[-1].append(input_word_index(vocabulary, token))
                    outputs[-1].append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
            if len(inputs[-1]) <= 1 or len(inputs[-1]) >= 200:
                #print("del\n", inputs[-1], '\n', outputs[-1])
                del inputs[-1]
                del outputs[-1]
                continue
            inputs[-1].append(input_word_index(vocabulary, "<END>"))
            outputs[-1].append(punctuation_index(punctuations, punctuation))
    return inputs, outputs


def save_to_pickle(inputs, outputs, vocabulary, punctuations, output_path):
    data = {"inputs": inputs, "outputs": outputs,
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

    inputs, outputs = words_to_ids(file_path, vocabulary, punctuations)
    print("Length of inputs is " + str(len(inputs)))
    assert len(inputs) == len(outputs)

    save_to_pickle(inputs, outputs, vocabulary, punctuations, output_path)

    EXAMPLES_PER_FILE = 500000 #TODO: Make it an parameter
    NUM_FILES = int(np.floor(len(inputs)/EXAMPLES_PER_FILE))
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        input = inputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        label = outputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
        print("Writing " + filename + " with length of " + str(len(input)) + " data.")
        example = tf.train.Example(features=tf.train.Features(feature={
            "inputs": _int64_feature(input),
            "labels": _int64_feature(label)}))
        writer.write(example.SerializeToString())
        writer.close()
    print("Converting Successfully.")


def make_example(sequence, labels):
    ex = tf.train.SequenceExample()
    #print(len(sequence))
    ex.context.feature["length"].int64_list.value.append(len(sequence))
    fl_inputs = ex.feature_lists.feature_list["inputs"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for input, label in zip(sequence, labels):
        fl_inputs.feature.add().int64_list.value.append(input)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def convert_file_according_sentences(file_path, vocabulary, punctuations, output_path):
    print("Converting " + file_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    inputs, outputs = sentences_to_ids(file_path, vocabulary, punctuations)
    inputs.sort(key=lambda x:len(x))
    outputs.sort(key=lambda x:len(x))
    print("Number of sentence is " + str(len(inputs)))
    assert len(inputs) == len(outputs)

    save_to_pickle(inputs, outputs, vocabulary, punctuations, output_path)

    SENTENCES_PER_FILE = 5000
    NUM_FILES = int(np.ceil(len(inputs)/SENTENCES_PER_FILE))
    for i in range(NUM_FILES):
        filename = os.path.join(output_path,  "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
        writer = tf.python_io.TFRecordWriter(filename)
        seqs = inputs[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        labs = outputs[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        print("Writing " + filename + " with " + str(len(seqs)) + " sentences.")
        for seq, label_seq in zip(seqs, labs):
            ex = make_example(seq, label_seq)
            writer.write(ex.SerializeToString())
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
    elif mode == "sentences":
        convert_file = convert_file_according_sentences

    convert_file(train_data, vocabulary, punctuations,
                os.path.join(data_path, "train"))
    convert_file(valid_data, vocabulary, punctuations,
                os.path.join(data_path, "valid"))
    convert_file(test_data, vocabulary, punctuations,
                os.path.join(data_path, "test"))


def main():
    convert_text_to_tfrecord(FLAGS.data_dir, conf=Conf(), mode="sentences", output_dir="sentence_data")


if __name__ == "__main__":
    main()
