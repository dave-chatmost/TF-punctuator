#!/bin/bash

# Created on 2017-07-12
# Author: Kaituo Xu
# Function: Chinese word segmentation with toolkit scws.
#   Download in http://www.xunsearch.com/scws/index.php

if [ $# != 3 ]; then
  echo "Chinese word segmentation with toolkit scws."
  echo "Usage: $0 <in-dir> <out-dir> <vocab>"
  exit 1;
fi

IN_DIR=$1
OUT_DIR=$2
VOCAB=$3

files=`ls $IN_DIR`

for file in $files; do 
  echo $IN_DIR/$file
  scws -i $IN_DIR/$file \
       -o $OUT_DIR/$file \
       -d $VOCAB
done
