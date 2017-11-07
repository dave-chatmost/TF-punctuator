#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu (Sogou)
# Function: Punctuate listed text files with LSTM model.
# NOTE: Execute in src directory.
echo $#

if [ $# != 9 ]; then
  echo "Punctuate many files with LSTM."
  echo "Usage: <in-dir> <out-dir> <model-dir> <model-config> <vocab> <punc-vocab> <blstm> <get-post> <GPU-ids>"
  echo "  blstm = true | false"
  echo "  get-post = True | False"
  exit 1;
fi

cd ../src

INPUT_DIR=$1
OUTPUT_DIR=$2
MODEL_DIR=$3
MODEL=$4
VOCAB=$5
PUNC_VOCAB=$6
BLSTM=$7
GET_POST=$8
GPU_IDS=$9

files=`cd $INPUT_DIR; ls *asr_out*`

[ ! -d $OUTPUT_DIR ] && mkdir -p $OUTPUT_DIR

if $BLSTM; then
    tool=punctuate_text_with_blstm.py
    LSTM=blstm
    echo "Punctuating with BLSTM"
else
    tool=punctuate_text_with_lstm.py
    LSTM=lstm
    echo "Punctuating with LSTM"
fi


for file in $files
do
    echo "Punctuating $file"
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python  $tool \
        --vocabulary=$VOCAB \
        --punct_vocab=$PUNC_VOCAB \
        --model=$MODEL --save_path=$MODEL_DIR \
        --log=log/punctuate_${file}_with_${LSTM}_$MODEL \
        --input_file=$INPUT_DIR/$file \
        --output_file=$OUTPUT_DIR/$file \
        --get_post=$GET_POST
    echo "Put the result in $OUTPUT_DIR/$file"
done
