import tensorflow as tf 

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data/data",
                    """Directory where exists text dataset""")
FLAGS = flags.FLAGS

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([FLAGS.data_dir])
_, serialized_example = reader.read(filename_queue)

sequence_features = {
    "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

_, sequence = tf.parse_single_sequence_example(
    serialized=serialized_example,
    sequence_features=sequence_features
)
inputs = sequence["inputs"]
labels = sequence["labels"]

batch_size=1
batch = tf.train.batch(
    tensors=[inputs, labels],
    batch_size=batch_size,
    dynamic_pad=True
)
input = batch[0]
label = batch[1]

init_state = tf.Variable(tf.zeros([], dtype=tf.int32))

num_steps=3
i = tf.train.range_input_producer(tf.size(input[0])//num_steps, shuffle=False).dequeue()
inp_block = tf.strided_slice(input, [0, i * num_steps],
                        [batch_size, (i + 1) * num_steps])
lab_block = tf.strided_slice(label, [0, i * num_steps],
                        [batch_size, (i + 1) * num_steps])


def test_parse_single_seq_ex():
    for i in range(8):
        seq, inp, lab = sess.run([sequence, inputs, labels])
        print(seq)
        print(inp, lab)

def test_batch():
    for i in range(3):
        bat = sess.run(batch)
        print(bat[0])
        print(bat[1])
        print("*"*20)

def test_block():
    state = sess.run(init_state)
    print(state)
    for idx in range(8):
        tempi, inp, lab = sess.run([i, inp_block, lab_block])
        state = state + 1
        if tempi == 0:
            state = sess.run(init_state)
        print(state)
        print(tempi)
        print(inp)
        print(lab)
        print("*"*20)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #test_parse_single_seq_ex()
    #test_batch()
    test_block()

    coord.request_stop()
    coord.join(threads)