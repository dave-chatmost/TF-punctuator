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
NUM_FILES = int(np.ceil(len(inputs)/EXAMPLES_PER_FILE))
EXAMPLES_PER_FEATURE = 2
# Make sure divisible
NUM_FEATURES = EXAMPLES_PER_FILE // EXAMPLES_PER_FEATURE
for i in range(NUM_FILES):
    filename = (PATH + "train.tfrecords-%.5d-of-%.5d" % (i+1, NUM_FILES))
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(NUM_FEATURES):
        if i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE > len(inputs):
            break
        input = inputs[i*EXAMPLES_PER_FILE + j*EXAMPLES_PER_FEATURE : i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE]
        label = outputs[i*EXAMPLES_PER_FILE + j*EXAMPLES_PER_FEATURE : i*EXAMPLES_PER_FILE + (j+1)*EXAMPLES_PER_FEATURE]
        print(input, label)
        example = tf.train.Example(features=tf.train.Features(feature={
            "inputs": _int64_feature(input),
            "labels": _int64_feature(label)}))
        writer.write(example.SerializeToString())
    writer.close()
