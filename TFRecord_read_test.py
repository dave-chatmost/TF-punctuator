import tensorflow as tf 

MATCH_FORMAT = "./data_test/train.tfrecords-*"
files = tf.train.match_filenames_once(MATCH_FORMAT)

filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        "inputs": tf.FixedLenFeature([2], tf.int64),
        "labels": tf.FixedLenFeature([2], tf.int64),
    })

input = tf.cast(features["inputs"], tf.int32)
label = tf.cast(features["labels"], tf.int32)

batch_size = 3
capacity = 1000 + 3 * batch_size

input_batch, label_batch = tf.train.shuffle_batch(
    [input, label], batch_size=batch_size, num_threads=2, 
    capacity=capacity, min_after_dequeue=30)


with tf.Session() as sess:
    # Required by tf.train.match_filenames_once()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        cur_input_batch, cur_label_batch = sess.run([input_batch, label_batch])
        print(cur_input_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)