#!/bin/bash

# Created on 2017-06-09
# Author: Kaituo Xu (Sogou)
# Funtion:
#	  Every line of the text file in this directory consists of
#	  <ASR output>^<reference transcript>, where ^ is delimiter.
#	  This script splits <ASR output> and <reference transcript>
#	  in differnt file.

dir=split
for file in biaozhu.txt \
        novel.txt \
        pm_punc_9531.txt \
        punc_test.txt \
        shu_ru_fa.long \
        shu_ru_fa.short \
        ted.txt.long \
        ted.txt.short \
        wang_yuquan.txt \
        xinhua_split.txt \
        xinhua.txt 
do
  echo "Spliting $file"
  awk 'BEGIN { FS="^" } { print $1 }' $file > ${dir}/${file}_asr_out
  awk 'BEGIN { FS="^" } { print $2 }' $file > ${dir}/${file}_ref
done
