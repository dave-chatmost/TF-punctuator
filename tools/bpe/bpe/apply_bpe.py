#!/bin/env python
# -*- coding: utf-8 -*-
# History:
#  v1.1, wangyuguang, fix encoding problem for chinese chars.
#  v1. Author: Rico Sennrich
"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
"""

from __future__ import unicode_literals, division

import re
import os
import sys
import time
import codecs
import argparse
import threading
from collections import defaultdict

# hack for python2/3 compatibility
#from io import open
argparse.open = open

import sys

# python 2/3 compatibility
if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

class BPE(object):

    def __init__(self, codes, separator='@@', lang='zh', skip_numbers=1):
        self.bpe_codes = [tuple(item.split()) for item in codes]
        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])
        self.separator = separator
        self.skip_numbers = skip_numbers
        self.lang = lang
        self.num = re.compile(r'^(\d+)$|(\d+\.\d+)$|(\d+(\,\d+)+)$|(\d+(\,\d+)+\.\d+)$')
        self.eng = re.compile(r'^[a-zA-Z\-\']+$', re.I)
        self.ascii = re.compile(r'^[!-~]+$', re.I)
        self.num_han = re.compile(r'^(\d+|\d+\.\d+)([^\d+a-zA-Z\.\,\']+)$', re.I)
        self.han_num = re.compile(r'^([^\d+a-zA-Z\.\,\']+)(\d+|\d+\.\d+)$', re.I)
        self.num_eng = re.compile(r'^(\d+|\d+\.\d+)([a-zA-Z\-\']+)$', re.I)
        self.eng_num = re.compile(r'^([a-zA-Z\-\']+)(\d+|\d+\.\d+)$', re.I)
        self.tag = re.compile(r'^\$[A-Z]+$')

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        output = []
        for word in sentence.split():
            # skip numbers
            if re.match(self.num, word):
                new_word = [word]
            elif re.match(self.tag, word):
                new_word = [word]
            elif word != '-' and "-" in word :
                new_word = [word.replace('-',' - ').strip()]
            elif re.match(self.ascii, word) and (self.lang == 'zh' or self.lang == 'ko' or self.lang == 'ja'):
                new_word = [word]
            elif re.match(self.num_eng, word):
                new_word = [re.sub(self.num_eng, '\g<1> \g<2>', word).strip()]
            elif re.match(self.eng_num, word):
                new_word = [re.sub(self.eng_num, '\g<1> \g<2>', word).strip()]
            elif re.match(self.num_han, word):
                new_word = [re.sub(self.num_han, '\g<1> \g<2>', word).strip()]
            elif re.match(self.han_num, word):
                new_word = [re.sub(self.han_num, '\g<1> \g<2>', word).strip()]
            else:
                new_word = encode(word, self.bpe_codes)

            if not self.skip_numbers:
                new_word = encode(word, self.bpe_codes)
            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])
        return ' '.join(output)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")
    parser.add_argument(
        '--lang', '-l', default='zh',
        help="Input text language [zh/en] (default: '%(default)s')")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--codes', '-c', metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--threads', '-t', type=int, default=1,
        help="Use multi threads to Apply BPE on texts")
    parser.add_argument(
        '--skip-numbers', '-sn', type=int, default=1,
        help="1: do not use bpe on numbers; 0: use bpe on numbers, default='%(default)s'")


    return parser

def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def encode(orig, bpe_codes, cache={}):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    word = tuple(orig) + ('</w>',)
    pairs = get_pairs(word)

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    cache[orig] = word
    return word

input_list = []
output_list = []
mutex = threading.Lock()
class worker(threading.Thread):
    def __init__(self, tid, thread_number, bpe):
        threading.Thread.__init__(self)
        self._tid = tid
        self._thread_number = thread_number
        self._bpe = bpe
        print >>sys.stderr,"work %d created"%(tid)

    def run(self):
        i=self._tid
        while (i<len(input_list)):
            cnt , orgn_line = input_list[i]
            if i%10000==0:
                print >>sys.stderr, "%s"%(i)
            # bpe segmentation
            out_line = self._bpe.segment(orgn_line).rstrip()
            #mutex.acquire()
            output_list[i] = out_line
            #mutex.release()
            i += self._thread_number

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.input = codecs.getreader('UTF-8')(args.input)
    args.output = codecs.getwriter('UTF-8')(args.output)

    codes = codecs.getreader('UTF-8')(open(args.codes, 'r'))
    bpe = BPE(codes, args.separator, lang=args.lang, skip_numbers=args.skip_numbers)
    for line in args.input:
        args.output.write(bpe.segment(line).rstrip())
        args.output.write('\n')

