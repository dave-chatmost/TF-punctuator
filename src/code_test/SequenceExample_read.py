import tensorflow as tf 

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data/data",
                    """Directory where exists text dataset""")
FLAGS = flags.FLAGS

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([FLAGS.data_dir])
_, serialized_example = reader.read(filename_queue)

context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

seq_length, sequence = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features
)
length = seq_length["length"]
inputs = sequence["inputs"]
labels = sequence["labels"]

batch_size=2
batch = tf.train.batch(
    tensors=[inputs, labels, length],
    batch_size=batch_size,
    dynamic_pad=True
)
input = batch[0]
label = batch[1]
seqlen = batch[2]

init_state = tf.Variable(tf.zeros([], dtype=tf.int32))


def test_parse_single_seq_ex():
    for i in range(8):
        seq, inp, lab, len = sess.run([sequence, inputs, labels, length])
        print(seq)
        print(inp, lab)
        print(len)

def test_batch():
    for i in range(3):
        bat = sess.run(batch)
        print(bat[0])
        print(bat[1])
        print(bat[2])
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
    test_batch()
    #test_block()

    coord.request_stop()
    coord.join(threads)