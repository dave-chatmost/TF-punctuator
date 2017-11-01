#!/bin/bash

# # Punctuate files with lstm
# #bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_lstm/hyp
# 
# # Char segment
# mkdir ~/punc-test-workspace/h300W_lstm/hyp_char
# python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_lstm/hyp ~/punc-test-workspace/h300W_lstm/hyp_char 
# 
# # Compute CER
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ~/punc-test-workspace/h300W_lstm/hyp_char asr_out
# 
# 
# #bash punc_many_files_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300w_helm/hyp exp/punc_data_head300w/5gram.arpa.bin
# #mkdir ~/punc-test-workspace/h300W_helm/hyp_char
# #python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_helm/hyp ~/punc-test-workspace/h300W_helm/hyp_char 
# #bash compute_wer.sh ~/punc-test-workspace/data/TN_char ~/punc-test-workspace/h300W_helm/hyp_char asr_out
# 
# bash get_post_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_helm/post ~/srilm-workspace/helm/exp/punc_data_head300W/5gram.arpa.bin 
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN/ ~/punc-test-workspace/h300W_lstm/post True
# punc_many_files_with_interpolate.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_helm/post/ ~/punc-test-workspace/h300W_lstm/post/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5/ 0.5
# mkdir ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char
# python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char/ asr_out
# 
# 
# bash punc_many_files_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/online5gram/hyp ~/data/online_punc_ngram/ai_merge.arpa2.binary ~/data/punct/vocab
# mkdir ~/punc-test-workspace/online5gram/hyp_char
# python ../tools/segment/charseg.py ~/punc-test-workspace/online5gram/hyp/ ~/punc-test-workspace/online5gram/hyp_char
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char/ ~/punc-test-workspace/online5gram/hyp_char/ asr_out


# hyp_dir=~/punc-test-workspace/h300W_lstm_hid8/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/h300W-hid8/model 0 hid8
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer


# hyp_dir=~/punc-test-workspace/h300W_lstm_hid9/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/h300W-hid9/model 1 hid9
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer

# hyp_dir=~/punc-test-workspace/TR/h300W_lstm_hid8/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/h300W-hid8/model 0 hid8 ../data/punc_data_head300W/vocab
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer

# hyp_dir=~/punc-test-workspace/TR/h300W_lstm_v3w_proj1/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/vocab3W-h300W-proj1/model 0 3wproj1 ../data/punc_data_head300W_vocab3W/vocab
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer
# 
# hyp_dir=~/punc-test-workspace/TR/h300W_lstm_v3w_hid8/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/vocab3W-h300W-hid8/model 0 3whid8 ../data/punc_data_head300W_vocab3W/vocab
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer
# 
# hyp_dir=~/punc-test-workspace/TR/h300W_lstm_v3w_hid256/hyp
# bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/vocab3W-h300W-hid256/model 0 3whid256 ../data/punc_data_head300W_vocab3W/vocab
# mkdir ${hyp_dir}_char
# python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
# bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer

hyp_dir=~/punc-test-workspace/TR/h300W_lstm_proj1/hyp
bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN $hyp_dir False ../exp/h300W-proj1/model 0 proj1 ../data/punc_data_head300W/vocab
mkdir ${hyp_dir}_char
python ../tools/segment/charseg.py $hyp_dir ${hyp_dir}_char 
bash compute_wer.sh ~/punc-test-workspace/data/TN_char ${hyp_dir}_char asr_out > ${hyp_dir%/*}/cer
