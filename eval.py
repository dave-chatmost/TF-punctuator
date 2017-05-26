import time
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

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
    _, valid_data, test_data = data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 256
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config,
                                  data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtest = PTBModel(is_training=False,
                                 config=eval_config, input_=test_input)

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            print("[DEBUG]" + ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                sv.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")
                return
            # QUEUE
            coord_test = tf.train.Coordinator()
            enqueue_threads_test = mtest._qr.create_threads(session, coord=coord_test, start=True)

            test_perplexity, predicts = run_epoch(session, mtest, verbose=True)
            logging.info("Test Perplexity: %.3f" % test_perplexity)

            # QUEUE
            coord_test.request_stop()
            coord_test.join(enqueue_threads_test)

        with open("predict.txt", "w") as f:
            f.write(str(predicts))
        labels = test_data["outputs"][:len(predicts)]
        precision, recall, fscore, support = score(labels, predicts)
        accuracy = accuracy_score(labels, predicts)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print('accuracy:', accuracy)


if __name__ == "__main__":
    tf.app.run()
