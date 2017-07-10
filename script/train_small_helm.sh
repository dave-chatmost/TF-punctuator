#!/usr/bin/bash

# Created on 2017-07-07
# Author: Kaituo Xu (Sogou)
# Funciton: Train Hidden Event Language Model (HELM), which is used to predict punctuation.

echo "$0 $@"
if [ $# != 3 ]; then
  echo "Usage: $0 <train-data-dir> <order> <LM-out-dir>"
  echo "NOTE: <train-data-dir> should include train.txt and vocab."
  echo "  default use modified Kneser-Ney discount and interpolate."
  exit 1;
fi

TRAIN_DATA=$1/train.txt
VOCAB=$1/vocab
ORDER=$2
LM=$3

[ ! -d $LM ] && mkdir -p $LM

ngram-count \
  -text $TRAIN_DATA \
  -vocab $VOCAB \
  -order $ORDER \
  -lm $LM/${ORDER}gram.arpa \
  -interpolate -kndiscount