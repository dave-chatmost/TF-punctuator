"""
Input:
    Unpunctuated text.
Output:
    Punctuated text.
"""

import logging

import tensorflow as tf

import punc_input
import utils
from conf import *
from model import *
from TFRecord_write import *

flags = tf.flags

flags.DEFINE_string("input_file", None,
                    "Where the unpunctuated text is stored (the text is already segmented).")
flags.DEFINE_string("output_file", "./punctuated_text",
                    "Where the punctuated text you want to put.")
flags.DEFINE_string("vocabulary", None,
                    "The same vocabulary used to train the LSTM model.")
flags.DEFINE_string("punct_vocab", None,
                    "The same punctution vocabulary used to train the LSTM model.")
flags.DEFINE_string("model", "test",
    "A type of model. Possible options are: small, medium, large, test.")
flags.DEFINE_string("save_path", None,
    "Model output directory.")
flags.DEFINE_string("log", "log",
    "Log filename.")
FLAGS = flags.FLAGS

logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')


def get_predicts(inputs, outputs):
    config = get_config(FLAGS.model)
    config.num_steps = 1
    config.batch_size = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        # Generate LSTM batch
        input_batch, label_batch = punc_input.eval_inputs("",
                                                          batch_size=config.batch_size,
                                                          inputs=inputs,
                                                          outputs=outputs)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtest = LSTMModel(input_batch=input_batch, label_batch=label_batch,
                              is_training=False, config=config)

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))

            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                logging.info("Model checkpoint file path: " + ckpt.model_checkpoint_path)
                sv.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                logging.info("No checkpoint file found")
                return

            epoch_size = len(inputs)

            test_perplexity, predicts = run_epoch(session, mtest, verbose=True, epoch_size=epoch_size)
            logging.info("Test Perplexity: %.3f" % test_perplexity)
        
        return predicts


def write_punctuations(input_file, predicts, punct_vocab_reverse_map, output_file):
    i = 0
    first_line = True 
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line in inpf:
            sentence_begin = True
            # print(line.split())
            for word in line.split():
                # print(word, i)
                punctuation = punct_vocab_reverse_map[predicts[i]]
                if sentence_begin and not first_line:
                    outf.write("%s\n%s" % (punctuation, word))
                    sentence_begin = False
                elif punctuation == " ":
                    outf.write("%s%s" % (punctuation, word))
                else:
                    outf.write(" %s %s" % (punctuation, word))
                i += 1
            first_line = False


def punctuator(input_file, vocab_file, punct_vocab_file, output_file):
    # Convert text to ids. (NOTE: fake outputs)
    vocabulary = load_vocabulary(vocab_file)
    punctuations = get_punctuations(punct_vocab_file)
    inputs, outputs = words_to_ids(input_file, vocabulary, punctuations)

    # Get predicts
    predicts = get_predicts(inputs, outputs)

    # Write punctuations
    punct_vocab_reverse_map = utils.get_reverse_map(punctuations)
    write_punctuations(input_file, predicts, punct_vocab_reverse_map, output_file)



if __name__ == "__main__":
    punctuator(FLAGS.input_file, FLAGS.vocabulary, FLAGS.punct_vocab, FLAGS.output_file)
