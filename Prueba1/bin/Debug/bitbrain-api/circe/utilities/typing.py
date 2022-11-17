# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:45:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

from functools import wraps
from typing import Optional, Union

from mne.io.brainvision.brainvision import RawBrainVision
from mne.io.array.array import RawArray



# %% TYPE-CHECKING DECORATORS
# We define some type-checking decorators.
# _____________________________________________________________________________

def type_check(func):
    """Checks the types of the arguments and return value and throws an error if there's a mismatch.
    NOTE: if you decorate a method with this decorator, then you need to call the object reference "self" """

    @wraps(func)
    def wrapper(*args, **kwargs):
        varnames = list(func.__code__.co_varnames)
        type_dict = func.__annotations__
        if varnames[0] == 'self':
            real_args = args[1:]
            varnames = varnames[1:]
        else:
            real_args = args
        for param, val in kwargs.items():
            type1 = type(val)
            try:
                type2 = type_dict[param]
            except KeyError:
                type2 = type1
            if type2 != type1:
                raise TypeError(f'Expected type {type2} for argument {param} '
                                f'but got type {type1}')
            varnames.remove(param)
        for param, val in zip(varnames, real_args):
            type1 = type(val)
            try:
                type2 = type_dict[param]
            except KeyError:
                type2 = type1
            if type2 != type1:
                raise TypeError(f'Expected type {type2} for argument {param} '
                                f'got type {type1}')
        result = func(*args, **kwargs)
        type1 = type(result)
        try:
            type3 = type_dict['return']
        except KeyError:
            type3 = type1
        if type3 != type1:
            raise TypeError(f'Expected return type {type3}  but got type '
                            f'{type1}')
        return result

    return wrapper


def class_type_check(class0):
    for key, val in vars(class0).items():
        if callable(val):
            setattr(class0, key, type_check(val))
    return class0



# %% SPECIAL TYPES
# We define special types.
# _____________________________________________________________________________

strn = Optional[str]
intn = Optional[int]
floatn = Optional[float]
listn = Optional[list]
dictn = Optional[dict]

fi = Union[float, int]
fin = Union[float, int, None]

ls = Union[list, str]
lsn = Union[list, str, None]
dlsn = Union[dict, list, str, None]

rawt = Union[RawBrainVision, RawArray]
