#!/bin/bash

# Created on 2017-09-25
# Author: Kaituo Xu (Sogou)
# Function: Punctuate listed text files with char LSTM model.
# NOTE: Execute in src directory.

if [ $# != 6 ]; then
  echo "Punctuate many files with LSTM."
  echo "Usage: <in-dir> <out-dir> <get-post> <model-dir> <GPU-ids> <model-config>"
  echo "  get-post = True | False"
  exit 1;
fi

cd ../src

INPUT_DIR=$1
OUTPUT_DIR=$2
GET_POST=$3
MODEL_DIR=$4
GPU_IDS=$5
MODEL=$6

files=`cd $INPUT_DIR; ls *asr_out*`

[ ! -d $OUTPUT_DIR ] && mkdir -p $OUTPUT_DIR

for file in $files
do
    echo "Processing $file"
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python punctuate_text_with_lstm.py \
        --vocabulary=../data/punc_data_head300W_char/vocab \
        --punct_vocab=../data/punc_data_head300W_char/punct_vocab \
        --model=$MODEL --save_path=$MODEL_DIR \
        --log=log/${MODEL}_punc_$file \
        --input_file=$INPUT_DIR/$file \
        --output_file=$OUTPUT_DIR/$file \
        --get_post=$GET_POST
    echo "Put the result in $OUTPUT_DIR/$file"
done
