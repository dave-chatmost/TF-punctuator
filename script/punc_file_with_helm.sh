#!/usr/bin/bash

# Created on 2017-07-07
# Author: Kaituo Xu (Sogou)
# Funciton: Predict punctuation for file using Hidden Event Language Model (HELM).

echo "$0 $@"
if [ $# != 5 ]; then
  echo "Usage: $0 <test-data> <helm> <order> <vocab-dir> <test-out>"
  echo "NOTE: <vocab-dir> should include vocab and hidden-vocab."
  exit 1;
fi

hidden-ngram \
  -text $1 \
  -lm $2 \
  -order $3 \
  -vocab $4/vocab \
  -hidden-vocab $4/hidden-vocab \
  -keep-unk > $5
