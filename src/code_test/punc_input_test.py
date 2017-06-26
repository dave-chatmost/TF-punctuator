import os
import sys
sys.path.append('/search/speech/xukaituo/tf_workspace/punctuator/src')

import tensorflow as tf

import punc_input

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data/data",
                    """Directory where exists text dataset""")
FLAGS = flags.FLAGS

input_batch, label_batch, files, i = punc_input.inputs(os.path.join(FLAGS.data_dir, "train"),
                                                    num_steps=3, batch_size=1,
                                                    tfrecords_format="tfrecords-00001-*",
                                                    mode="sentences")

with tf.Session() as sess:
    # Required by tf.train.match_filenames_once()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for x in range(8):
        cur_input_batch, cur_label_batch, cur_i = sess.run([input_batch, label_batch, i])
        print(cur_input_batch)
        print(cur_label_batch)
        print(cur_i)
    coord.request_stop()
    coord.join(threads)
