"""
Useful utils for this project.
"""

import tensorflow as tf


def count_number_trainable_params():
    """ 
    Counts the number of trainable variables.
    """
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        # print(shape)
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    """
    Computes the total number of params for a given shape.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    """
    nb_params = 1
    for dim in shape:
        nb_params = nb_params * int(dim)
    return nb_params

def get_reverse_map(dictionary):
    return {v:k for k,v in dictionary.items()}

