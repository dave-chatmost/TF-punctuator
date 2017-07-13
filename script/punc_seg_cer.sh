#!/bin/bash

# Punctuate files with lstm
#bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_lstm/hyp

# Char segment
mkdir ~/punc-test-workspace/h300W_lstm/hyp_char
python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_lstm/hyp ~/punc-test-workspace/h300W_lstm/hyp_char 

# Compute CER
bash compute_wer.sh ~/punc-test-workspace/data/TN_char ~/punc-test-workspace/h300W_lstm/hyp_char asr_out


#bash punc_many_files_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300w_helm/hyp exp/punc_data_head300w/5gram.arpa.bin
#mkdir ~/punc-test-workspace/h300W_helm/hyp_char
#python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_helm/hyp ~/punc-test-workspace/h300W_helm/hyp_char 
#bash compute_wer.sh ~/punc-test-workspace/data/TN_char ~/punc-test-workspace/h300W_helm/hyp_char asr_out

bash get_post_with_helm.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_helm/post ~/srilm-workspace/helm/exp/punc_data_head300W/5gram.arpa.bin 
bash punc_many_files_with_lstm.sh ~/punc-test-workspace/data/TN/ ~/punc-test-workspace/h300W_lstm/post True
punc_many_files_with_interpolate.sh ~/punc-test-workspace/data/TN ~/punc-test-workspace/h300W_helm/post/ ~/punc-test-workspace/h300W_lstm/post/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5/ 0.5
mkdir ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char
python ../tools/segment/charseg.py ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char
bash compute_wer.sh ~/punc-test-workspace/data/TN_char/ ~/punc-test-workspace/h300W_helm_h300W_lstm/inter.5_char/ asr_out
