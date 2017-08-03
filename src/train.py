import logging
import os

import numpy as np
import tensorflow as tf

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


def train():
    """ Train Punctuator for a number of epochs."""
    config = get_config(FLAGS.model)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        input_batch, label_batch, mask_batch, files = punc_input.inputs(
            os.path.join(FLAGS.data_path, "train"),
            num_steps=config.num_steps,
            batch_size=config.batch_size)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = LSTMModel(input_batch=input_batch, label_batch=label_batch,
                          mask_batch=mask_batch, is_training=True, config=config)
        tf.summary.scalar("Training_Loss", m.cost)
        tf.summary.scalar("Learning_Rate", m.lr)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))
            logging.info(session.run(files))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            epoch_size = punc_input.get_epoch_size(FLAGS.data_path + "/train.pkl",
                                                   config.batch_size, config.num_steps)
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True,
                                             epoch_size=epoch_size)
                logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            coord.request_stop()
            coord.join(threads)

            if FLAGS.save_path:
                logging.info("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                              global_step=sv.global_step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.DeleteRecursively(FLAGS.save_path)
    tf.gfile.MakeDirs(FLAGS.save_path)
    train()


if __name__ == "__main__":
    tf.app.run()
