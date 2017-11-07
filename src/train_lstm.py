import argparse
import logging
import os

import numpy as np
import tensorflow as tf

import punc_input_lstm
import utils
from conf import *
from model import lstm

parser = argparse.ArgumentParser(description='LSTM punctuation prediction training')
# data
parser.add_argument('--train_data', default='',
                    help='Training tfrecord data path.')
parser.add_argument('--cv_data', default='',
                    help='Cross validation tfrecord data path.')
# model hyper parameters
parser.add_argument('--init_scale', default=0.1, type=float,
                    help='Init range of model parameters.')
parser.add_argument('--vocab_size', default=100000+2, type=int,
                    help='Input vocab size. (Include <UNK> and <END>)')
parser.add_argument('--embedding_size', default=256, type=int,
                    help='Input embedding size.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='Hidden size of LSTM.')
parser.add_argument('--proj_size', default=256, type=int,
                    help='Projection size of LSTM.')
parser.add_argument('--hidden_layers', default=3, type=int,
                    help='Number of LSTM layers')
parser.add_argument('--num_class', default=5, type=int,
                    help='Number of output classes. (Include blank space " ")')
parser.add_argument('--max_norm', default=5, type=int,
                    help='Norm cutoff to prevent explosion of gradients')
# training hyper parameters
parser.add_argument('--bptt_step', default=20, type=int,
                    help='Truncated BPTT step.')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=2, type=int,
                    help='Number of training epochs')
parser.add_argument('--start_decay_epoch', default=2, type=int,
                    help='Epoch start to decay learning rate')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    help='Initial learning rate')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay rate')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
# logging
parser.add_argument('--log', default='temp.log',
                    help='Log filename')


def main(args):
    """ Train Punctuator for a number of epochs."""
    utils.makedir(args.save_folder)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -args.init_scale, args.init_scale)

        input_batch, label_batch, mask_batch, files = punc_input_lstm.inputs(
            os.path.join(args.train_data, "train"),
            num_steps=args.bptt_step,
            batch_size=args.batch_size)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = lstm.LSTMPunctuator(input_batch=input_batch, label_batch=label_batch,
                                    mask_batch=mask_batch, is_training=True,
                                    vocab_size=args.vocab_size,
                                    embedding_size=args.embedding_size,
                                    hidden_size=args.hidden_size,
                                    proj_size=args.proj_size,
                                    hidden_layers=args.hidden_layers,
                                    num_class=args.num_class,
                                    max_norm=args.max_norm,
                                    batch_size=args.batch_size,
                                    bptt_step=args.bptt_step
                                    )
        tf.summary.scalar("Training_Loss", m.cost)
        tf.summary.scalar("Learning_Rate", m.lr)

        sv = tf.train.Supervisor(logdir=args.save_folder)
        with sv.managed_session() as session:
            logging.info(args)
            logging.info("Number of parameters: {}".format(utils.count_number_trainable_params()))
            logging.info(session.run(files))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            epoch_size = punc_input_lstm.get_epoch_size(args.train_data + "/train.pkl",
                                                        args.batch_size, args.bptt_step)
            for i in range(args.epochs):
                lr_decay = args.lr_decay ** max(i + 1 - args.start_decay_epoch, 0.0)
                m.assign_lr(session, args.lr * lr_decay)
                logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity = lstm.run_epoch(session, m, eval_op=m.train_op, verbose=True,
                                                  epoch_size=epoch_size)
                logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            coord.request_stop()
            coord.join(threads)

            if args.save_folder:
                logging.info("Saving model to %s." % args.save_folder)
                sv.saver.save(session, args.save_folder,
                              global_step=sv.global_step)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, filemode='w',
                        level=logging.INFO,
                        format='[%(levelname)s %(asctime)s] %(message)s')
    main(args)
