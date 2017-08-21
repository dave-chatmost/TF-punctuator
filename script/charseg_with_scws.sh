#!/bin/bash

# Created on 2017-07-12
# Author: Kaituo Xu
# Function: Chinese char segmentation with toolkit scws.
#   Download in http://www.xunsearch.com/scws/index.php

if [ $# != 2 ]; then
  echo "Chinese char segmentation with toolkit scws."
  echo "Usage: $0 <in-dir> <out-dir>"
  exit 1;
fi

IN_DIR=$1
OUT_DIR=$2

files=`ls $IN_DIR`

[ ! -d $OUT_DIR ] && mkdir -p $OUT_DIR

for file in $files; do 
  echo $IN_DIR/$file
  scws -i $IN_DIR/$file \
       -o $OUT_DIR/$file
done
