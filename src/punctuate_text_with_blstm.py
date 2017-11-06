"""
Input:
    Unpunctuated text.
Output:
    Punctuated text.
"""

import logging

import tensorflow as tf

import punc_input_blstm
import utils
from conf import *
from model import *
from convert_text_to_TFRecord import *

flags = tf.flags

flags.DEFINE_string("input_file", None,
                    "Where the unpunctuated text is stored (the text is already segmented).")
flags.DEFINE_string("output_file", "./punctuated_text",
                    "Where the punctuated text you want to put.")
flags.DEFINE_boolean("get_post", False,
                    "Get punctuation posteriors.")
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

def inference_sentences_to_ids(file_path, vocabulary, punctuations):
    inputs = []
    outputs = []
    lens = []
    
    with open(file_path, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            # Skip blank line
            if len(line.strip()) == 0:
                continue
            punctuation = " "
            inputs.append([])
            outputs.append([])
            for token in line.split():
                if token in punctuations:
                    punctuation = token
                else:
                    inputs[-1].append(input_word_index(vocabulary, token))
                    outputs[-1].append(punctuation_index(punctuations, punctuation))
                    punctuation = " "
            inputs[-1].append(input_word_index(vocabulary, "<END>"))
            outputs[-1].append(punctuation_index(punctuations, punctuation))
            lens.append(int(len(inputs[-1])))
    return inputs, outputs, lens


def get_predicts(inputs, outputs, lens, get_post=False):
    config = get_config(FLAGS.model)
    config.num_steps = 1
    config.batch_size = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        # Generate LSTM batch
        input_batch, label_batch, len_batch = punc_input_blstm.eval_inputs(
            inputs=inputs,
            outputs=outputs,
            lens=lens)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtest = LSTMModel(input_batch=input_batch, label_batch=label_batch,
                              seq_len=len_batch, is_training=False, config=config)

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

            epoch_size = len(inputs) #// config.batch_size

            test_perplexity, predicts = run_epoch(session, mtest, verbose=True, epoch_size=epoch_size,
                                                  get_post=get_post, debug=False)
            logging.info("Test Perplexity: %.3f" % test_perplexity)
        
        return predicts


def write_punctuations(input_file, predicts, punct_vocab_reverse_map, output_file):
    print(predicts)
    with open(input_file, 'r', encoding='utf8') as inpf, open(output_file, 'w', encoding='utf8') as outf:
        i = 0
        for line in inpf:
            j = 0
            for word in line.split():
                punctuation = punct_vocab_reverse_map[predicts[i][j]]
                if punctuation == " ":
                    outf.write("%s " % (word))
                else:
                    outf.write("%s %s " % (punctuation, word))
                j += 1
            # <END>
            punctuation = punct_vocab_reverse_map[predicts[i][j]]
            outf.write("%s\n" % (punctuation))
            i += 1


def write_posteriors(input_file, posteriors, punct_vocab_reverse_map, output_file):
    punct_vocab_reverse_map[0]="*noevent*"
    i = 0
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line in inpf:
            new_sentence = True
            for word in line.split():
                if new_sentence:
                    new_sentence = False
                    i += 1
                outf.write("%s\t" % word)
                for j in range(len(punct_vocab_reverse_map)):
                    outf.write(" %s %f" % (punct_vocab_reverse_map[j], posteriors[i][j]))
                outf.write("\n")
                i += 1
            # <END>
            #punctuation = punct_vocab_reverse_map[predicts[i]]
            #outf.write("%s\n" % (punctuation))
            #i += 1


def punctuator(input_file, vocab_file, punct_vocab_file, output_file, get_post):
    # Convert text to ids. (NOTE: fake outputs)
    vocabulary = load_vocabulary(vocab_file)
    punctuations = get_punctuations(punct_vocab_file)
    punct_vocab_reverse_map = utils.get_reverse_map(punctuations)
    inputs, outputs, lens = inference_sentences_to_ids(input_file, vocabulary, punctuations)

    # Get predicts
    if get_post:
        posteriors = get_predicts(inputs, outputs, lens, get_post)
        write_posteriors(input_file, posteriors, punct_vocab_reverse_map, output_file)
        return 
    else:
        predicts = get_predicts(inputs, outputs, lens)

    # Write punctuations
    write_punctuations(input_file, predicts, punct_vocab_reverse_map, output_file)



if __name__ == "__main__":
    punctuator(FLAGS.input_file, FLAGS.vocabulary, FLAGS.punct_vocab, FLAGS.output_file, FLAGS.get_post)
