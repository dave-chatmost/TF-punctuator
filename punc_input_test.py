import os

import tensorflow as tf

import punc_input

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../punc_data/data",
                    """Directory where exists text dataset""")
FLAGS = flags.FLAGS

input_batch, label_batch, files = punc_input.inputs(os.path.join(FLAGS.data_dir, "train"),
                                                    num_steps=20, batch_size=2)

with tf.Session() as sess:
    # Required by tf.train.match_filenames_once()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        cur_input_batch, cur_label_batch = sess.run([input_batch, label_batch])
        print(cur_input_batch)
        print(cur_label_batch)
    coord.request_stop()
    coord.join(threads)
