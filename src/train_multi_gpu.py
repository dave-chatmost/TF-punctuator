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
flags.DEFINE_string("tblog", "tblog",
    "Tensorboard log dir"
)
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use."""
)

FLAGS = flags.FLAGS

logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
    """ Train Punctuator for a number of epochs."""
    config = get_config(FLAGS.model)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        epoch_size = punc_input.get_epoch_size(FLAGS.data_path + "/data/train.pkl",
                                               config.batch_size, config.num_steps)
        epoch_size = epoch_size // FLAGS.num_gpus
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
        lr = tf.train.exponential_decay(config.learning_rate,
                                        global_step,
                                        epoch_size,
                                        config.lr_decay,
                                        staircase=True
                                        )
        opt = tf.train.GradientDescentOptimizer(lr)

        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        input_batch, label_batch, files = punc_input.inputs(os.path.join(FLAGS.data_path, "data/train"),
                                                            num_steps=config.num_steps,
                                                            batch_size=config.batch_size)

        tower_grads = []
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('GPU_%d' % i) as scope:
                        m = LSTMModel(input_batch=input_batch, label_batch=label_batch,
                                      is_training=True, config=config)
                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(m.cost)
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables. 
        # That operation also increments `global_step`.
        train_op = opt.apply_gradients(grads, global_step=global_step)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as session:
            init = tf.global_variables_initializer()
            tf.local_variables_initializer().run()
            session.run(init)

            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))
            logging.info(session.run(files))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            summary_writer = tf.summary.FileWriter(FLAGS.tblog, session.graph)
            for i in range(config.max_max_epoch):
                logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(lr)))
                train_perplexity = run_epoch(session, m, eval_op=train_op, verbose=True,
                                             epoch_size=epoch_size, summary_writer=summary_writer, 
                                             num_gpus=FLAGS.num_gpus)
                logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                if FLAGS.save_path:
                    logging.info("Saving model to %s." % FLAGS.save_path)
                    saver.save(session, FLAGS.save_path,
                                global_step=i+1)
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)


#        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
#        with sv.managed_session() as session:
#            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))
#            logging.info(session.run(files))
#
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(sess=session, coord=coord)
#            summary_writer = tf.summary.FileWriter(FLAGS.tblog, session.graph)
#            for i in range(config.max_max_epoch):
#                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
#                m.assign_lr(session, config.learning_rate * lr_decay)
#                logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
#
#                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True,
#                                             epoch_size=epoch_size, summary_writer=summary_writer)
#                logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
#            summary_writer.close()
#
#            coord.request_stop()
#            coord.join(threads)
#
#            if FLAGS.save_path:
#                logging.info("Saving model to %s." % FLAGS.save_path)
#                sv.saver.save(session, FLAGS.save_path,
#                              global_step=sv.global_step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.DeleteRecursively(FLAGS.save_path)
    tf.gfile.MakeDirs(FLAGS.save_path)
    if tf.gfile.Exists(FLAGS.tblog):
        tf.gfile.DeleteRecursively(FLAGS.tblog)
    tf.gfile.MakeDirs(FLAGS.tblog)
    train()


if __name__ == "__main__":
    tf.app.run()
