# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:00:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries and modules.
# _____________________________________________________________________________

from circe.utilities.typing import class_type_check, strn, intn, floatn, \
                                   listn, dictn, fi, fin, ls, lsn, dlsn



# %% CLUSTERING FUNCTIONS
# We define several clustering functions.
# _____________________________________________________________________________

def GetGrid(n):
    n_sqrt = n ** 0.5
    n_sqr = int(n_sqrt)
    if n_sqrt - n_sqr == 0:
        w = n_sqr
        h = n_sqr
    else:
        if n_sqr * (n_sqr + 1) > n:
            w = n_sqr + 1
            h = n_sqr
        else :
            w = n_sqr + 1
            h = n_sqr + 1
    return w, h
