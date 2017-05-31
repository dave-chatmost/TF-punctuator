import inspect
import logging
import time

import numpy as np
import tensorflow as tf


def run_epoch(session, model, eval_op=None, verbose=False, epoch_size=1):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    predicts = []

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "logits": model.logits
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    logging.info(epoch_size) 
    for step in range(epoch_size):
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
        iters += model.num_steps

        if verbose and step % (epoch_size // 100) == 10:
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    if eval_op is None:
        # Make the predicts right format
        final_predicts = np.concatenate(
            np.array(predicts).reshape([-1, model.batch_size]).T,
            axis=0).tolist()
        return np.exp(costs / iters), final_predicts
    else:
        return np.exp(costs / iters)


class LSTMModel(object):
    """The Punctuation Prediction LSTM Model."""

    def __init__(self, input_batch, label_batch, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps

        hidden_size = config.hidden_size
        num_proj = config.num_proj
        vocab_size = config.vocab_size
        punc_size = config.punc_size

        # Set LSTM cell
        def lstm_cell():
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.LSTMCell(
                    hidden_size, use_peepholes=True, num_proj=num_proj,
                    forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)],
            state_is_tuple=True)
        
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # Embedding part
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_batch)
        
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Define output
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        if num_proj is not None:
            hidden_size = num_proj

        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, punc_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [punc_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._logits = tf.nn.softmax(logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(label_batch, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        # End __init__ 

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def logits(self):
        return self._logits
