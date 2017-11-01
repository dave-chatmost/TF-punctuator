#!/usr/bin/env python3

import collections
import os
import sys
from collections import OrderedDict


def read_words(filename, encoding='gbk', unit='word'):
    data = []
    with open(filename, 'r', encoding=encoding, errors="ignore") as inf:
        for line in inf:
            if unit == 'char':
                for char in line.strip():
                    if char != " ":
                        data.append(char)
            else:
                for word in line.strip().split():
                    data.append(word)
    return data


def build_vocab(corpus, output_file, encoding, unit):
    data = read_words(corpus, encoding, unit)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words = list(zip(*count_pairs))[0]

    with open(output_file, "w", encoding=encoding) as vocab:
        for i in range(len(words)):
            vocab.write(words[i] + "\n")

def build_vocab2(corpus, output_file, encoding, unit="words"):
    word_freqs = {}
    with open(corpus, 'r', encoding=encoding, errors="ignore") as inf:
        i = 0
        for line in inf:
            for w in line.strip().split():
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1
            i += 1
            if i % 10000 == 0:
                print("Processed {} sentences".format(i))
    sort_by_freq = sorted(word_freqs.items(), key=lambda x: (-x[1], x[0]))
    words = list(zip(*sort_by_freq))[0]

    with open(output_file, "w", encoding=encoding) as vocab:
        for i in range(len(words)):
            vocab.write(words[i] + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_vocab.py <corpus> <outfile> <encoding> <unit>")
        print("e.g.:python generate_vocab.py train.txt vocab utf-8 word")
        print("encoding = utf-8 | GBK")
        print("unit = char | word (not support char right now)")
        exit(0)
    build_vocab2(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
