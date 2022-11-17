# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 13:00:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

import os
import pickle
import pandas as pd
import json

from circe.utilities.typing import intn


# %% FILE-RELATED FUNCTIONS
# We define functions for saving and loading objects.
# _____________________________________________________________________________

def CheckPath(path: str,
              to_be: str):
    if to_be == 'folder':
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise TypeError('\'path\' should be folder.')
    elif to_be == 'file':
        if not os.path.isfile(path):
            raise TypeError('\'path\' should be file.')
    else:
        raise ValueError('\'to_be\' must be \'folder\' or \'file\'.')

def LoadCSV(path: str,
            sep: str = ',',
            dec: str = '.',
            n_rows: intn = None,
            columns = None,
            header: intn = None):
    CheckPath(path=path, to_be='file')
    if header == None:
        header = 'infer'
    data = pd.read_csv(path, sep=sep, decimal=dec, nrows=n_rows,
                       usecols=columns, header=header)
    print('.csv file loaded from {}.'.format(path))
    return data

def LoadJSON(path: str):
    CheckPath(path=path, to_be='file')
    with open(path) as file:
            data = json.load(file)
    print('.json file loaded from {}.'.format(path))
    return data

def SaveCSV(path: str,
            data,
            sep: str = ',',
            dec: str = '.',
            index: bool = False):
    data.to_csv(path, sep=sep, decimal=dec, index=index)
    print('.csv file saved to \'{}\'.'.format(path))

def SavePKL(obj, path: str):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
    print('.pkl file saved to \'{}\'.'.format(path))

def LoadPKL(path: str):
    print("mirar aqui:::"+path)
    CheckPath(path=path, to_be='file')
    obj = pickle.load(open(path, 'rb'))
    print('.pkl file loaded from \'{}\'.'.format(path))
    return obj
