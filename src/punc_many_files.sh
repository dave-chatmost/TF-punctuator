#!/bin/bash

INPUT_DIR=../ngram_test_data/TN
OUTPUT_DIR=../ngram_test_data/hyp

MODEL=proj1

files="biaozhu.txt_asr_out_tn"
for file in $files
do
    CUDA_VISIBLE_DEVICES=0 \
    python punctuate_text_with_lstm.py \
        --vocabulary=../punc_data_head300W/vocab \
        --punct_vocab=../punc_data_head300W/punct_vocab \
        --model=$MODEL --save_path=../exp/h300W-$MODEL/model \
        --log=log/punc_text_test \
        --input_file=$INPUT_DIR/$file \
        --output_file=$OUTPUT_DIR/$file
done