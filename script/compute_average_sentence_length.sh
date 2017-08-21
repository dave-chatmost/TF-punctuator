#!/bin/bash

# Created on 2017-07-10
# Author: Kaituo Xu (Sogou)
# Funtion: Get posteriors for each word with HELM.

echo "$0 $@"

if [ $# != 1 ]; then
  echo "Compute average sentence length of each file, given a directory."
  echo "Usage: <in-dir>"
  exit 1;
fi

IN_DIR=$1

files=`ls $IN_DIR`
for file in $files; do
  word=`cat $IN_DIR/$file | awk 'BEGIN{ sum=0 } { sum += NF } END{ print sum }' | awk '{print $1}'`
  line=`wc -l $IN_DIR/$file | awk '{print $1}'`
  echo "$file $((word/line))" 
done