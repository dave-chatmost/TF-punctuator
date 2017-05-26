import time
import numpy as np 
import tensorflow as tf 

import reader
from conf import *
from model import *


flags = tf.flags
#logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large, test."
)
flags.DEFINE_string(
    "data_path", None,
    "Where the training/test data is stored."
)
flags.DEFINE_string(
    "save_path", None,
    "Model output directory."
)
flags.DEFINE_string(
    "log", "log",
    "Log filename."
)

FLAGS = flags.FLAGS

import logging
logging.basicConfig(filename=FLAGS.log, filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s %(asctime)s] %(message)s')


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (len(data["inputs"]) // batch_size) // num_steps # don't need to minus 1 anymore
        # self.epoch_size = ((len(data) // batch_size) - 1) // num_steps # the output of LM is shift by 1, so minus 1
        # ********* IO ********** #
        # print("[DEBUG]", self.batch_size, self.num_steps, self.epoch_size, len(data), len(data["inputs"]))
        self.input_data, self.targets, self.qr = reader.batch_producer(
            data, batch_size, num_steps, name=name)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    predicts = []

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "logits": model._logits
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    logging.info(model.input.epoch_size) 
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h 
        
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        if eval_op is None:
            predict = vals["logits"]
            predicts.extend(np.argmax(predict, 1).tolist())

        costs += cost
        iters += model.input.num_steps

        # logging.info(step, model.input.epoch_size // 10, step % (model.input.epoch_size // 10))
        # if verbose and step % (model.input.epoch_size // 10) == 10:
        if verbose and step % (model.input.epoch_size // 100) == 10:
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    # Make the predicts right format
    predicts = np.concatenate(
        np.array(predicts).reshape([-1, model.input.batch_size]).T,
        axis=0
    ).tolist()
    # if eval_op is None:
    #     logging.info(predicts.tolist())
    return np.exp(costs / iters), predicts


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    if FLAGS.model == "small2":
        return SmallConfig2()
    if FLAGS.model == "small3":
        return SmallConfig3()
    elif FLAGS.model == "medium":
        return MediumConifg()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    # ********* IO ********** #
    data = reader.load_data(raw_data_path=FLAGS.data_path, conf=DataConf())
    train_data, valid_data, test_data = data

    config = get_config()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(
                config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config,
                             input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(
                config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False,
                                  config=config, input_=valid_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            # QUEUE
            coord_train = tf.train.Coordinator()
            coord_valid = tf.train.Coordinator()
            enqueue_threads_train = m._qr.create_threads(session, coord=coord_train, start=True)
            enqueue_threads_valid = mvalid._qr.create_threads(session, coord=coord_valid, start=True)

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                logging.info("Epoch: %d Learning rate: %.3f" %
                      (i + 1, session.run(m.lr)))

                train_perplexity, _ = run_epoch(
                    session, m, eval_op=m.train_op, verbose=True)
                logging.info("Epoch: %d Train Perplexity: %.3f" %
                      (i + 1, train_perplexity))

                valid_perplexity, _ = run_epoch(session, mvalid)
                logging.info("Epoch: %d Valid Perplexity: %.3f" %
                      (i + 1, valid_perplexity))

            # QUEUE
            coord_train.request_stop()
            coord_train.join(enqueue_threads_train)
            coord_valid.request_stop()
            coord_valid.join(enqueue_threads_valid)

            if FLAGS.save_path:
                logging.info("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
