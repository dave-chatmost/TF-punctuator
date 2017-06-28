import tensorflow as tf
from random import randrange

# fake data
#sequences = [[1,2,3], [4,5,6,7], [8,9]]
#label_sequences = [[2, 1, 1], [2, 3, 1, 1], [4, 3]]
sequences = []
label_sequences = []
num_seq=20
for i in range(num_seq):
    sequences.append([])
    label_sequences.append([])
    for e in range(randrange(5,20)):
        sequences[-1].append(randrange(1,9))
        label_sequences[-1].append(randrange(1,5))
for i in range(num_seq):
    print(sequences[i], label_sequences[i])
print("Above is fake data.")

print("Sort by sequence length")
sequences.sort(key=lambda x:len(x))
label_sequences.sort(key=lambda x:len(x))
for i in range(num_seq):
    print(sequences[i], label_sequences[i])
print("Above is sorted data.")


def make_example(sequence, labels):
    ex = tf.train.SequenceExample()
    ex.context.feature["length"].int64_list.value.append(len(sequence))
    fl_inputs = ex.feature_lists.feature_list["inputs"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for input, label in zip(sequence, labels):
        fl_inputs.feature.add().int64_list.value.append(input)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def write_one_file():
    filename = 'Records/output.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for seq, label_seq in zip(sequences, label_sequences):
        print(seq, label_seq)
        ex = make_example(seq, label_seq)
        writer.write(ex.SerializeToString())
    writer.close()


def write_many_file():
    SENTENCES_PER_FILE = 5
    NUM_FILES = len(sequences)//SENTENCES_PER_FILE
    for i in range(NUM_FILES):
        filename = 'Records/tfrecords-%.5d-of-%.5d' % (i+1, NUM_FILES)
        writer = tf.python_io.TFRecordWriter(filename)
        seqs = sequences[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        labs = label_sequences[i*SENTENCES_PER_FILE: (i+1)*SENTENCES_PER_FILE]
        for seq, label_seq in zip(seqs, labs):
            print(seq, label_seq)
            ex = make_example(seq, label_seq)
            writer.write(ex.SerializeToString())
        print("*"*20)
        writer.close()


if __name__ == "__main__":
    write_many_file()