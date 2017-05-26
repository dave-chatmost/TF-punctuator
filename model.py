import inspect
import tensorflow as tf


class PTBModel(object):
    """The PTB LSTM Model."""

    def __init__(self, is_training, config, input_, inputs, labels):
        self._input = input_
        self._qr = input_.qr

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        num_proj = config.num_proj
        vocab_size = config.vocab_size
        punc_size = config.punc_size

        # Set LSTM cell
        def basic_lstm_cell():
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True)

        def lstm_cell():
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.LSTMCell(
                    hidden_size, use_peepholes=True, num_proj=num_proj,
                    forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)

        attn_cell = lstm_cell
        # attn_cell = basic_lstm_cell
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
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        
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
            [tf.reshape(input_.targets, [-1])],
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
    def input(self):
        return self._input

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

