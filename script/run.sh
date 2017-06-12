#!/bin/bash

# Created on 2017-06-08
# Author: Kaituo Xu (Sogou Inc.)
# Funciton: Simple example for running this project.
# NOTE: Execute in src directory.

cd ../src
# Train
# CUDA_VISIBLE_DEVICES=0 --> Set the GPU ID 
CUDA_VISIBLE_DEVICES=0 python train.py --data_path=../punc_data_head300W/ --save_path=../exp/h300W-hid7/model --model=hid7 --log=log/h300W-hid7 &

# Evaluate
CUDA_VISIBLE_DEVICES=0 python eval.py --data_path=../punc_data_head300W/ --save_path=../exp/h300W-hid7/model --model=hid7 --log=log/h300W-hid7-eval &
