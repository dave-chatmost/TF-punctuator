import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

import punc_input
import utils
from conf import *
from model import *

flags = tf.flags

flags.DEFINE_string("model", "test",
    "A type of model. Possible options are: small, medium, large, test."
)
flags.DEFINE_string("data_path", None,
    "Where the training/test data is stored."
)
flags.DEFINE_string("save_path", None,
    "Model output directory."
)
flags.DEFINE_string("log", "log",
    "Log filename."
)

FLAGS = flags.FLAGS

logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')


def evaluate():
    """Evaluate punctuator."""
    config = get_config(FLAGS.model)
    config.num_steps = 1
    config.batch_size = 128

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        input_batch, label_batch, seq_len, files = punc_input.inputs(os.path.join(FLAGS.data_path, "test"),
                                                                     batch_size=config.batch_size, fileshuf=False,
                                                                     mode="sentences")

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            mtest = LSTMModel(input_batch=input_batch, label_batch=label_batch,
                              seq_len=seq_len, is_training=False, config=config)

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            logging.info(session.run(files))
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))

            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                logging.info("Model checkpoint file path: " + ckpt.model_checkpoint_path)
                sv.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                logging.info("No checkpoint file found")
                return

            #epoch_size = 100
            epoch_size = punc_input.get_epoch_size(FLAGS.data_path + "/test.pkl",
                                                   config.batch_size, config.num_steps,
                                                   EXAMPLES_PER_FILE=1)

            test_perplexity, predicts = run_epoch(session, mtest, verbose=True, epoch_size=epoch_size, debug=False)
            logging.info("Test Perplexity: %.3f" % test_perplexity)

        logging.info("predicts' length = {}".format(len(predicts)))
        pred_file = os.path.join(FLAGS.save_path, "predict.txt")
        with open(pred_file, "w") as f:
            for i in range(len(predicts)):
                f.write(str(predicts[i]) + '\n')
            #f.write(str(predicts) + '\n')

        test_data=np.load(FLAGS.data_path + "/test.pkl")
        labels = test_data["outputs"][:len(predicts)]

        label_file = os.path.join(FLAGS.save_path, "label.txt")
        with open(label_file, "w") as f:
            for i in range(len(labels)):
                f.write(str(labels[i]) + '\n')
            #f.write(str(labels) + '\n')

        l = []
        p = []
        for i in range(len(predicts)):
            if len(predicts[i]) != len(labels[i]):
                print(i,'\t', len(predicts[i])-len(labels[i]))
                continue
                #print(predicts[i])
                #print(labels[i])
            l.extend(labels[i])
            p.extend(predicts[i])
        labels = l
        predicts = p

        precision, recall, fscore, support = score(labels, predicts)#score(predicts, labels)
        accuracy = accuracy_score(labels, predicts)
        logging.info('precision: {}'.format(precision))
        logging.info('recall: {}'.format(recall))
        logging.info('fscore: {}'.format(fscore))
        logging.info('support: {}'.format(support))
        logging.info('accuracy: {}'.format(accuracy))


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
