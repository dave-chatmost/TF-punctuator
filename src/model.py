import inspect
import logging
import time

import numpy as np
import tensorflow as tf


def run_epoch(session, model, eval_op=None, verbose=False, epoch_size=1, num_gpus=1):
    """Runs the model on the given data."""
    start_time = time.time()
    all_words = 0
    costs = 0.0
    predicts = []

    fetches = {
        "cost": model.cost,
        "mask": model.mask,
        "predict": model.predicts,
        "seqlen": model.seq_len
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    logging.info("Epoch size: %d" % epoch_size) 
    for step in range(epoch_size):
        vals = session.run(fetches)
        cost = vals["cost"]
        mask = vals["mask"]
        if eval_op is None:
            predict = vals["predict"]
            if step > 497:
                #for i in range(len(mask)):
                #    print(mask[i])
                print(np.sum(mask, axis=1))
                print(vals["seqlen"])
            mask = np.array(np.round(mask), dtype=np.int32)
            shape = mask.shape
            if step > 10 and step < 20:
                print(predict)
                #print(np.argmax(predict, 1))
            predict = np.reshape(np.argmax(predict, 1), shape).tolist()
            mask = np.sum(mask, axis=1).tolist()
            for i in range(shape[0]):
                predicts.append(predict[i][:mask[i]])
            #predicts.extend(np.argmax(predict, 1).tolist())

        costs += cost
        all_words += np.sum(mask)

        if epoch_size < 100:
            verbose = False

        if verbose and step % (epoch_size // 100) == 10:
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / step),
                   num_gpus * all_words / (time.time() - start_time)))

    if eval_op is None:
        # Make the predicts right format
        #final_predicts = np.concatenate(
        #    np.array(predicts).reshape([-1, model.batch_size]).T,
        #    axis=0).tolist()
        return np.exp(costs / epoch_size), predicts
    else:
        return np.exp(costs / epoch_size)


class LSTMModel(object):
    """The Punctuation Prediction LSTM Model."""

    def __init__(self, input_batch, label_batch, seq_len, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps

        embedding_size = config.embedding_size
        hidden_size = config.hidden_size
        num_proj = config.num_proj
        vocab_size = config.vocab_size
        punc_size = config.punc_size

        # Set LSTM cell
        def lstm_cell():
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
        
        # Embedding part
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, embedding_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_batch)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Automatically reset state in each batch
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=inputs,
                                               sequence_length=seq_len,
                                               dtype=tf.float32)

        if num_proj is not None:
            hidden_size = num_proj

        output = tf.reshape(outputs, [-1, hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, punc_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [punc_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._predicts = tf.nn.softmax(logits)

        # Generate mask matrix to mask loss
        maxlen = tf.cast(tf.reduce_max(seq_len), tf.int32) # it can not work on type int64
        ones = tf.ones([maxlen, maxlen], dtype=tf.float32)
        low_triangular_ones = tf.matrix_band_part(ones, -1, 0)
        mask = tf.gather(low_triangular_ones, seq_len-1)
        self._mask = tf.cast(mask, tf.int32)
        self.seq_len = seq_len
        mask_flat = tf.reshape(mask, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.reshape(label_batch, [-1])
        )
        masked_loss = mask_flat * loss
        self._cost = cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask_flat)

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = self._grads = tf.clip_by_global_norm(tf.gradients(cost, tvars),
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
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def predicts(self):
        return self._predicts

    @property
    def grads(self):
        return self._grads

    @property
    def mask(self):
        return self._mask
