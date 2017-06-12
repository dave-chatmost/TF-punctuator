#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu (Sogou)
# Function: Remove English letter, number and punctuation. 

INDIR=../ngram_test_data/segment
OUTDIR=../ngram_test_data/TN
files=`ls $INDIR`

for file in $files
do
  echo "Processing $file"
  sed -e 's/[[:alnum:][:punct:]] //g' -e 's/[[:alnum:][:punct:]] *//g' $INDIR/$file > $OUTDIR/${file}_tn
done
