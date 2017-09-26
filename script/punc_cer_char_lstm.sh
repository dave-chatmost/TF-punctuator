#!/bin/bash

hyp_dir=~/punc-test-workspace/char_lstm_proj1/hyp_char
bash punc_many_files_with_char_lstm.sh ~/punc-test-workspace/data/TN_char $hyp_dir False ../exp/char-300W-proj1/model 7 proj1-char
bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir} asr_out > ${hyp_dir%/*}/cer
