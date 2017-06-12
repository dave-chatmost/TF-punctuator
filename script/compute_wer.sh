#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu (Sogou)
# Function: Compute WER for listed hyp files.
# NOTE: You need to download Kaldi at first, because
#       we use kaldi utils: compute-wer.

(
cd ~/tools/kaldi/kaldi/egs/wsj/s5/
. ./path.sh
)

REF_DIR=../ngram_test_data/TN
HYP_DIR=../ngram_test_data/hyp

files=`ls $HYP_DIR/*asr_out*`
for hyp in $files
do
  echo -e "\n*****\nFILE: $hyp\n*****"
  cor_ref=$REF_DIR/`basename ${hyp/asr_out/ref}`
  cor_unpunc=$REF_DIR/`basename $hyp`
  # Add utt-id to per utterance
  awk '{ print NR "\t" $0 }' $hyp > temp_hyp
  awk '{ print NR "\t" $0 }' $cor_ref > temp_ref
  awk '{ print NR "\t" $0 }' $cor_unpunc > temp_unpunc
  echo "---------- BEFORE PUNCTUATION ----------"
  compute-wer --text --mode=present ark:temp_ref ark:temp_unpunc
  echo ""
  echo "---------- AFTER PUNCTUATION ----------"
  compute-wer --text --mode=present ark:temp_ref ark:temp_hyp
done
rm -f temp_hyp temp_ref temp_unpunc
