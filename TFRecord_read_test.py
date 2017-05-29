import tensorflow as tf

MATCH_FORMAT = "./data_test/tfrecords-*"
files = tf.train.match_filenames_once(MATCH_FORMAT)

filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

EXAMPLES_PER_FILE = 6
features = tf.parse_single_example(
    serialized_example,
    features={
        "inputs": tf.FixedLenFeature([EXAMPLES_PER_FILE], tf.int64),
        "labels": tf.FixedLenFeature([EXAMPLES_PER_FILE], tf.int64),
    })

input = features["inputs"]
label = features["labels"]

num_steps = 2
batch_size = 4 # variable option
data_len = tf.size(input)
batch_len = tf.to_int32(data_len / batch_size)

input_data = tf.transpose(
                tf.reshape(
                    tf.slice(input, [0], [batch_size * batch_len]), 
                    [batch_size, batch_len]))
label_data = tf.transpose(
                tf.reshape(
                    tf.slice(label, [0], [batch_size * batch_len]), 
                    [batch_size, batch_len]))

capacity = 3 * batch_size

input_batch, label_batch = tf.train.batch(
    [input_data, label_data], batch_size=num_steps, num_threads=2, 
    capacity=capacity, enqueue_many=True, shapes=[[batch_size], [batch_size]])

input_batch = tf.transpose(input_batch)
label_batch = tf.transpose(label_batch)

with tf.Session() as sess:
    # Required by tf.train.match_filenames_once()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #print(sess.run([input, label]))
    #print(sess.run([data_len, batch_len]))
    #print(sess.run([input_data, label_data]))
    #print()
    for i in range(6):
        cur_input_batch, cur_label_batch = sess.run([input_batch, label_batch])
        print(cur_input_batch)
        print(cur_label_batch)
    coord.request_stop()
    coord.join(threads)
