#!/bin/bash

# Created on 2017-07-11
# Author: Kaituo Xu (Sogou)

if [ $# != 5 ]; then
  echo "Punctuate many files by interpolating LSTM with HELM."
  echo "Usage: <in-dir> <post-dir1> <post-dir2> <out-dir> <weight>"
  exit 1;
fi

INPUT_DIR=$1
POST_DIR1=$2
POST_DIR2=$3
OUTPUT_DIR=$4
WEIGHT=$5

files=`cd $INPUT_DIR; ls *asr_out*`

[ ! -d $OUTPUT_DIR ] && mkdir -p $OUTPUT_DIR

for file in $files; do
  python punc_many_files_with_interpolate.py \
    $INPUT_DIR/$file \
    $POST_DIR1/$file \
    $POST_DIR2/$file \
    $OUTPUT_DIR/$file \
    $WEIGHT
done
