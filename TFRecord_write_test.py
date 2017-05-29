import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Fake inputs and outputs
inputs = list(range(0, 20))
outputs = list(range(20, 40))
print(inputs)
print(outputs)

PATH = "./data_test/"
if tf.gfile.Exists(PATH):
    tf.gfile.DeleteRecursively(PATH)
tf.gfile.MakeDirs(PATH)


EXAMPLES_PER_FILE = 6
NUM_FILES = int(np.floor(len(inputs)/EXAMPLES_PER_FILE))
for i in range(NUM_FILES):
    filename = PATH + "tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES)
    writer = tf.python_io.TFRecordWriter(filename)
    input = inputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
    label = outputs[i*EXAMPLES_PER_FILE : (i+1)*EXAMPLES_PER_FILE]
    print("Writing " + filename + " with length of " + str(len(input)) + " data.")
    example = tf.train.Example(features=tf.train.Features(feature={
        "inputs": _int64_feature(input),
        "labels": _int64_feature(label)}))
    writer.write(example.SerializeToString())
    writer.close()
print("Converting Successfully.")
