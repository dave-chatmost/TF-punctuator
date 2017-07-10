#!/usr/bin/bash

# Created on 2017-07-08
# Author: Kaituo Xu
# Function: Train big Hidden Event Language Model (HELM).

echo "$0 $@"
if [ $# != 3 ]; then
  echo "Usage: $0 <train-data-dir> <order> <LM-out-dir>"
  echo "NOTE: <train-data-dir> should include train.txt and vocab."
  echo "  default use modified Kneser-Ney discount and interpolate."
  exit 1;
fi

TRAIN_DATA=`pwd`/$1/train.txt
VOCAB=`pwd`/$1/vocab
ORDER=$2
LM=`pwd`/$3
TMP=$3/tmp

[ ! -d $TMP ] && mkdir -p $TMP
cd $TMP

# Step 1: Split big train data.
mkdir split
split -l 100000 $TRAIN_DATA split/
ls split | sed "s:^:split/:" > splitlist

# Step 2: Count counts of split train data.
mkdir counts
make-batch-counts splitlist 5 cat counts -order $ORDER

# Step 3: Merge counts.
merge-batch-counts counts

# Step 4: Train Big LM
make-big-lm -read counts/*.ngrams.gz \
  -vocab $VOCAB -unk -order $ORDER \
  -interpolate -kndiscount \
  -lm $LM/${ORDER}gram.arpa

#rm -rf $TMP