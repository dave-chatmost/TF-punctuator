#!/usr/bin/env python

import os
import sys


def max_match(sentence, dictionary):
    """Word segmentation in Chinese: the maximum matching algorithm.
    Args:
        sentence: Chinese character string.
        dictionary: dict of Chinese words.
    Return:
        Chinese words sequence list.
    """

    if len(sentence) == 0: # empty sentence
        return []
    
    for i in range(len(sentence), 0, -1): # len(sentence), ... , 1
        firstword = sentence[:i]
        remainder = sentence[i:]
        if dictionary.get(firstword):
            seq = [firstword]
            seq.extend(max_match(remainder, dictionary))
            return seq
    
    # no word was found, so make a one-character word
    # In GBK, one byte for 00-7F, two bytes for others
    if sentence[0] <= '\x7f':
        word_bytes_num = 1
    else:
        word_bytes_num = 2
    firstword = sentence[:word_bytes_num]
    remainder = sentence[word_bytes_num:]
    seq = [firstword]
    seq.extend(max_match(remainder, dictionary))
    return seq


def max_match_test():
    sentence = "ilovepythonhowaboutyou" # i love python how about you
    dictionary = {"i":1, "we":1, "you":1, "your":1, "love":1, "python":1, "how":1, "about":1, "what":1, "lo":1, "yo":1}
    word_seq = max_match(sentence, dictionary)
    print(word_seq)
    print((" ").join(word_seq))


def word_seg(text_file_list_file, dict_file, out_dir):
    """Do word segmentaion using maximum matching algorithm.
    Steps:
        1. load dictionary
        2. read text file
        3. for each line in text:
            seg_result.append(max_match(line, dictionary))
        4. write seg_result to file
    Args:
        text_file_list_file: a file contain a list of text file path.
        dict_file: dictionary file path.
        out_dir: put the segmented file here.
    """
    def _load_dict(dict_file):
        with open(dict_file, 'r') as f:
            dict = {word.strip('\n') : 1 for word in f}
        return dict
    
    dictionary = _load_dict(dict_file)
    with open(text_file_list_file, 'r') as list:
        for file in list:
            file = file.strip()
            out_file = os.path.join(out_dir, os.path.basename(file))
            print("Segmenting", file)
            print("Put the segmented file into", out_file)
            with open(file, 'r') as text, open(out_file, 'w') as result:
                for line in text:
                    sentence = line.strip()
                    seg_sentence = max_match(sentence, dictionary)
                    result.write(" ".join(seg_sentence) + "\n")


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Usage: python wordseg.py <text-file-list-file> <dictionary-file>")
        sys.exit(-1)

    text_file_list_file = sys.argv[1]
    dict_file = sys.argv[2]
    out_dir = os.path.join(os.getcwd(), "segment")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    word_seg(text_file_list_file, dict_file, out_dir)
