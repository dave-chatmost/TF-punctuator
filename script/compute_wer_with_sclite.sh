#!/bin/bash

# Created on 2017-07-13
# Author: Kaituo Xu (Sogou)
# Function: Compute WER for hyp files that match pattern.
# NOTE: You need to download sctk at first, because
#       we use sctk utils: sclite

echo "$0 $@"

if [ $# -le 2 ]; then
  echo "Compute WER for hyp files that match pattern."
  echo "Usage: $0 <ref-dir> <hyp-dir> <simple-file-pattern>"
  echo "e.g.: $0 data/TN data/hyp asr_out"
  echo "NOTE: When use this script, you MUST add sctk/bin to your PATH."
  exit 1
fi

REF_DIR=$1
HYP_DIR=$2
PATTERN=$3

# In order to use `sclite`
export PATH=$PATH:~/tools/kaldi/tools/sctk/bin

files=`ls $HYP_DIR/*$PATTERN*`
for hyp in $files; do
  echo "*****"
  echo "FILE: $hyp"
  echo "*****"
  cor_ref=$REF_DIR/`basename ${hyp/$PATTERN/ref}`
  cor_unpunc=$REF_DIR/`basename $hyp`
  # Add utt-id to per utterance
  awk '{ print $0 "(" NR "_V)" }' $hyp > temp_hyp
  awk '{ print $0 "(" NR "_V)" }' $cor_ref > temp_ref
  awk '{ print $0 "(" NR "_V)" }' $cor_unpunc > temp_unpunc
  echo "---------- BEFORE PUNCTUATION ----------"
  sclite -r temp_ref trn -h temp_unpunc trn $hyp -i spu_id | grep 'Err\|Sum'
  echo ""
  echo "---------- AFTER PUNCTUATION ----------"
  sclite -r temp_ref trn -h temp_hyp trn $hyp -i spu_id | grep 'Err\|Sum'
done
rm -f temp_hyp temp_ref temp_unpunc
