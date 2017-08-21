#!/bin/bash

# Created on 2017-07-13
# Author: Kaituo Xu (Sogou)
# Function: Get results of different interpolate weight.

# bash get_post_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/online5gram/post ~/data/online_punc_ngram/ai_merge.arpa2.binary ~/data/punct/vocab

result=~/punc-test-workspace/online_all_lstm/result.txt

[ -f $result ] && rm $result

for i in {1..9}; do
  echo "[ Weight is 0.$i --> ngram : lstm = $i : $((10-i)) ]" >> $result
  # Interpolate
  bash punc_many_files_with_interpolate.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/online5gram/post/ ~/punc-test-workspace/all_lstm/post/ ~/punc-test-workspace/online_all_lstm/inter.$i 0.$i
  # Char segment
  mkdir ~/punc-test-workspace/online_all_lstm/inter.${i}_char
  python ../tools/segment/charseg.py ~/punc-test-workspace/online_all_lstm/inter.${i} ~/punc-test-workspace/online_all_lstm/inter.${i}_char/
  # Compute cer
  bash compute_wer.sh ~/punc-test-workspace/data/TN_char/ ~/punc-test-workspace/online_all_lstm/inter.${i}_char/ asr_out >> $result
  echo -e '\n\n' >> $result
done

echo "Put CER result in $result"