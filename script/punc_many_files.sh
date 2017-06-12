#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu (Sogou)
# Function: Punctuate listed text files with LSTM model.
# NOTE: Execute in src directory.

cd ../src

INPUT_DIR=../ngram_test_data/TN
OUTPUT_DIR=../ngram_test_data/hyp
MODEL=proj1
files=`cd $INPUT_DIR; ls *asr_out*`

for file in $files
do
    echo "Processing $file"
    CUDA_VISIBLE_DEVICES=0 \
    python punctuate_text_with_lstm.py \
        --vocabulary=../punc_data_head300W/vocab \
        --punct_vocab=../punc_data_head300W/punct_vocab \
        --model=$MODEL --save_path=../exp/h300W-$MODEL/model \
        --log=log/punc_$file \
        --input_file=$INPUT_DIR/$file \
        --output_file=$OUTPUT_DIR/$file
    echo "Put the result in $OUTPUT_DIR/$file"
done
