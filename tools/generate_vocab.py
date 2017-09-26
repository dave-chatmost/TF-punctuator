#!/usr/bin/env python3

import collections
import os
import sys


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

    with open(output_file, "w", encoding='gbk') as vocab:
        for i in range(len(words)):
            vocab.write(words[i] + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_vocab.py <corpus> <outfile> <encoding> <unit>")
        print("e.g.:python generate_vocab.py train.txt vocab utf-8 word")
        print("encoding = utf-8 | GBK")
        print("unit = char | word")
        exit(0)
    build_vocab(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
