#!/usr/bin/env python

import sys


def max_match(sentence, dictionary):
    """Word segmentation in Chinese: the maximum matching algorithm.
    Args:
        sentence: Chinese character string.
        dictionary: a list of Chinese words.
    Return:
        Chinese words sequence list.
    """

    if len(sentence) == 0: # empty sentence
        return []
    
    for i in range(len(sentence), 0, -1): # len(sentence), ... , 1
        firstword = sentence[:i]
        remainder = sentence[i:]
        if firstword in dictionary:
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
    dictionary = ["i", "we", "you", "your", "love", "python", "how", "about", "what", "lo", "yo"]
    word_seq = max_match(sentence, dictionary)
    print(word_seq)
    print((" ").join(word_seq))


def word_seg(text_file_list_file, dict_file):
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
    """
    def _load_dict(dict_file):
        with open(dict_file, 'r') as f:
            dict = [word.strip('\n') for word in f]
        return dict
    
    dictionary = _load_dict(dict_file)
    # print(dictionary)
    with open(text_file_list_file, 'r') as list:
        for file in list:
            file = file.strip()
            print("Segmenting", file)
            with open(file, 'r') as text, open(file+"_seg", 'w') as result:
                for line in text:
                    # print(line.strip('\n'))
                    sentence = line.strip()
                    # print(max_match(sentence, dictionary))
                    result.write(" ".join(max_match(sentence, dictionary)) + "\n")


if __name__ == "__main__":
    text_file_list_file = sys.argv[1]
    dict_file = sys.argv[2]
    word_seg(text_file_list_file, dict_file)
