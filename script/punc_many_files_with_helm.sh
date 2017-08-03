#!/bin/bash

# Created on 2017-07-09
# Author: Kaituo Xu (Sogou)
# Funtion: Punctuate many files with Hidden Event Language Model.

echo "$0 $@"

if [ $# != 4 ]; then
  echo "Punctuate many files with Hidden Event Language Model."
  echo "Usage: <in-dir> <out-dir> <helm> <vocab>"
  exit 1;
fi

IN_DIR=$1
OUT_DIR=$2
LM=$3
VOCAB=$4

files=`ls $IN_DIR/*asr*`

echo "Step 1: Put all unpunctuated files together, record their lines and names."
i=0
for file in $files; do
  line=`wc -l $file | awk '{print $1}'`
  lines[i]=$line
  names[i]=${file##*/}
  echo ${lines[i]} ${names[i]}
  i=$((i+1))
  cat $file | awk '{ print "<s> " $0 " </s>" }' >> all_asr_out
done

echo "Step 2: Punctuating with very big HELM"
hidden-ngram -text all_asr_out -order 5 -vocab $VOCAB -hidden-vocab ~/data/online_punc_ngram/hidden-vocab -keep-unk -lm $LM > all_hyp_helm

echo "Step 3: Split punctuated file"
[ ! -d $OUT_DIR ] && mkdir $OUT_DIR
processed=0
for j in `seq 0 $((i-1))`; do
  processed=$(($processed+${lines[$j]}))
  head -$processed all_hyp_helm | tail -${lines[j]} | awk '{ $1=""; $NF=""; print $0 }' > $OUT_DIR/${names[j]}
done

rm all_asr_out
