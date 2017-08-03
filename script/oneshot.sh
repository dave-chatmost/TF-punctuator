#!/bin/bash

# Step 1: Word segmentation.    -->    tools/wordseg/wordseg.py
#       Usage: python wordseg.py <text-file-list-file> <dictionary-file> <out-dir>
# Step 2: Simple text Normalization.    -->    TN.sh 
#       Usage: specify the INDIR and OUTDIR in this script.
# Step 3: Punctuate text file.    -->    punc_many_files.sh
# Step 4: Compute WER.    -->    compute_wer.sh

file=$1

echo `pwd`/$file > tmplist
python ../tools/segment/wordseg.py tmplist ~/data/punct/vocab ./tmpseg

mv ./tmpseg/$1 ./tmpseg/asr_out
bash punc_many_files_with_lstm.sh `pwd`/tmpseg/ `pwd`/tmppunc/ False ../exp/all-proj1/model 0
cat ./tmppunc/asr_out | tr -d ' ' > ${file}_punc

rm -rf tmplist ./tmpseg ./tmppunc
