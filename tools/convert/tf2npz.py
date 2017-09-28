#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Shanbo Cheng: chengshanbo@sogou-inc.com
#
# Python release: Anaconda 2.7
#
# Date: 2017-07-03 20:14:07
# Last modified: 2017-07-28 11:01:18

"""
Convert tensorflow model to npz
"""

import sys
import numpy
import tensorflow as tf

if len(sys.argv) != 3:
    sys.stderr.write('usage: python %s <model.ckpt> <output>\n' % __file__)
    sys.stderr.write('e.g.: python %s model.ckpt-5940 model.npz\n' % __file__)
    sys.exit(-1)


checkpoint = sys.argv[1]
var_list = tf.contrib.framework.list_variables(checkpoint)
reader = tf.contrib.framework.load_checkpoint(checkpoint)

var_dict = {}
shape_dict = {}

def process_embedding(value):
    # print(value.shape)
    # print(value[0][0])
    # print(value[1][0])
    # print(value[-2][0])
    # print(value[-1][0])
    value[[0, -2], :] = value[[-2, 0], :] # switch first and <unk>
    value[[1, -1], :] = value[[-1, 1], :] # switch second and <END>
    # value[0], value[-2] = value[-2], value[0] 
    # value[1], value[-1] = value[-1], value[1]
    # print(value[0][0])
    # print(value[1][0])
    # print(value[-2][0])
    # print(value[-1][0])
    return value

def cut_name(name):
    cut = name.split('/')
    # print(cut)
    if len(cut) == 2: # embedding, softmax_b, softmax_w
        name = cut[1]
    elif len(cut) == 6: # cell parameter excluding projection
        name = "layer" + cut[3][-1] + "/" + cut[5]
    elif len(cut) == 7:
        name = "layer" + cut[3][-1] + "/" + cut[5] + "/" + cut[6]
    return name

print("Reading tensorflow model")
for (name, shape) in var_list:
    #print(len(shape))
    if len(shape) == 0: continue
    value = reader.get_tensor(name)
    if name == "Model/embedding":
        print(value[0][0])
        print(value[1][0])
        print(value[-2][0])
        print(value[-1][0])
        value = process_embedding(value)
        print(value[0][0])
        print(value[1][0])
        print(value[-2][0])
        print(value[-1][0])
    name = cut_name(name)
    # print(name)
    var_dict[name] = value
    shape_dict[name] = shape
val = float(input("Input number of layers: "))
num_layer = numpy.zeros((1, 1))
num_layer[0][0] = val
var_dict['num_layer'] = num_layer.astype('float32') 
shape_dict['num_layer'] = num_layer.shape 


print("Saving npz model")
for (k, v) in var_dict.items():
    print(k, v.shape)
numpy.savez(sys.argv[2], **var_dict)

# for (k, v) in shape_dict.items():
# 	print(k, v)

print("Done")
