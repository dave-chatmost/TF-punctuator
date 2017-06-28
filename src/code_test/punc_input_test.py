import os
import sys
sys.path.append('/search/speech/xukaituo/tf_workspace/TF-punctuator/src')

import tensorflow as tf

import punc_input

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data/data",
                    """Directory where exists text dataset""")
FLAGS = flags.FLAGS

input_batch, label_batch, seq_len, files = punc_input.inputs(os.path.join(FLAGS.data_dir, "train"),
                                                    num_steps=3, batch_size=256,
                                                    #tfrecords_format="tfrecords-00030-*",
                                                    mode="sentences")
maxlen = tf.cast(tf.reduce_max(seq_len), tf.int32) # it can not work on type int64
ones = tf.ones([maxlen, maxlen], dtype=tf.float32)
low_triangular_ones = tf.matrix_band_part(ones, -1, 0)
mask = tf.gather(low_triangular_ones, seq_len-1)                                                    

with tf.Session() as sess:
    # Required by tf.train.match_filenames_once()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for x in range(20):
        cur_input_batch, cur_label_batch, cur_seq_len, cur_mask = sess.run([input_batch, label_batch, seq_len, mask])
        print(cur_input_batch)
        print(cur_label_batch)
        print(cur_seq_len)
        print(cur_mask)

    coord.request_stop()
    coord.join(threads)
