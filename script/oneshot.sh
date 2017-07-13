#!/bin/bash

# Step 1: Word segmentation.    -->    tools/wordseg/wordseg.py
#       Usage: python wordseg.py <text-file-list-file> <dictionary-file> <out-dir>
# Step 2: Simple text Normalization.    -->    TN.sh 
#       Usage: specify the INDIR and OUTDIR in this script.
# Step 3: Punctuate text file.    -->    punc_many_files.sh
# Step 4: Compute WER.    -->    compute_wer.sh

file=$1

cat $file > tmplist
python ../../tools/segment/wordseg.py list ~/data/punct/vocab ./tmpseg
