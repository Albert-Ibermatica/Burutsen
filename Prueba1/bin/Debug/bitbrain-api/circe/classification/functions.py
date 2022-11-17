# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:00:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries and modules.
# _____________________________________________________________________________

import tensorflow as tf

from sklearn.feature_selection import SelectKBest, f_classif, f_regression



# %% CLASSIFICATION FUNCTIONS
# We define several classification functions.
# _____________________________________________________________________________

def SelectKFeats(data, feats, target, k, select_func='f_class'):
    if type(select_func) != str:
        raise TypeError('\'select_func\' must be string.')
    if select_func == 'f_class':
        select_func_call = f_classif
    elif select_func == 'f_regress':
        select_func_call = f_regression
    else:
        raise ValueError('{} is not a supported function.' \
                         .format(select_func))
    feats0 = SelectKBest(score_func=select_func_call, k=k) \
             .fit(data[feats], data[target]) \
             .get_feature_names_out() \
             .tolist()
    feats_str = ', '.join(feats0)
    print('{}/{} selected features: {}.'.format(k, len(feats),
                                                feats_str))
    return feats0

def DataSet(data, feats, target, shuffle=True, batch_size=32, seed=1):
    if type(target) == str:
        data = data.copy()[feats+[target]]
        labels = data.pop(target)
        dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    elif target is None:
        data = data.copy()[feats]
        dataset = tf.data.Dataset.from_tensor_slices(dict(data))
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=len(data), seed=seed)
    dataset = dataset.batch(batch_size)
    return dataset
