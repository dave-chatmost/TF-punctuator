#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu (Sogou)
# Function: Remove English letter, number, punctuation and blank line. 

if [ $# -lt 2 ]; then
  echo "Remove English letter, number, punctuation and blank line."
  echo "Usage: $0 <input-dir> <output-dir>"
  echo "e.g.: $0 data/segment data/TN"
  exit 1;
fi

INDIR=$1
OUTDIR=$2

# TODO: when processing GBK file, there is a encoding problem.
# source kaldi path.sh can solve this problem.
# solve this problem later.
cd ~/tools/kaldi/kaldi/egs/wsj/s5/; . ./path.sh; cd -

files=`ls $INDIR`
for file in $files; do
  echo "Processing $file"
  sed -e 's/[[:alnum:][:punct:]] //g' \
  -e 's/[[:alnum:][:punct:]] *//g' \
  -e '/^ *$/d' \
  $INDIR/$file  > $OUTDIR/${file}_tn
  echo "Put the result in " $OUTDIR/${file}_tn
done