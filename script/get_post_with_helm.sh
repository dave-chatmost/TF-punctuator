#!/bin/bash

# Created on 2017-07-10
# Author: Kaituo Xu (Sogou)
# Funtion: Get posteriors for each word with HELM.

echo "$0 $@"

if [ $# != 4 ]; then
  echo "Get punctuation posteriors with Hidden Event Language Model."
  echo "Usage: <in-dir> <out-dir> <helm> <vocab>"
  exit 1;
fi

IN_DIR=$1
OUT_DIR=$2
LM=$3
VOCAB=$4

echo "Step 1: Put all unpunctuated files together, record their words and names."
files=`ls $IN_DIR/*asr*`
i=0
for file in $files; do
  word=`cat $file | awk 'BEGIN{ sum=0 } { sum += NF } END{ print sum }' | awk '{print $1}'`
  #word=`wc -w $file | awk '{print $1}'`
  line=`wc -l $file | awk '{print $1}'`
  words[i]=$(($word+2*$line))
  names[i]=${file##*/}
  #words[1]=1035715 # There are some errors with novel.txt_asr_out_tn
  echo ${words[i]} ${names[i]}
  let i++
  cat $file | awk '{ print "<s> " $0 " </s>" }' >> all_asr_out
done

echo "Step 2: Get posteriors with very big HELM"
hidden-ngram -text all_asr_out -order 5 -vocab $VOCAB -hidden-vocab ~/data/online_punc_ngram/hidden-vocab -keep-unk -lm $LM -posteriors > all_post_helm

echo "Step 3: Split posteriors file"
[ ! -d $OUT_DIR ] && mkdir $OUT_DIR
processed=0
for j in `seq 0 $((i-1))`; do
  processed=$(($processed+${words[$j]}))
  head -$processed all_post_helm | tail -${words[j]} | grep -v '\(<s>\)\|\(</s>\)' > $OUT_DIR/${names[j]}
done

rm all_asr_out