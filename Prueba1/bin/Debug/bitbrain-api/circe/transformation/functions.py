
# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

import numpy as np

import pandas as pd
from pandas.core.frame import DataFrame, Series

from scipy.fft import fft
from scipy.signal import spectrogram

from pywt import downcoef



# %% DATA-RELATED FUNCTIONS
# We define functions for performing data transformations.
# _____________________________________________________________________________

def Group(data, feats, funcs, consts, fft_n, dwt_mode, spect_freq):
    data_grouped = {}
    feats_grouped = []
    for feat in feats:
        for func in funcs:
            feat_grouped = feat + '_' + func.upper()
            if func in ['fft', 'spect'] or func.startswith('dwt_'):
                if func == 'fft':
                    transf = np.abs(fft(x=data[feat].values, n=fft_n))
                elif func.startswith('dwt_'):
                    wavelet = func.split('_')[1]
                    transf = downcoef(part='a',
                                      data=data[feat].values,
                                      wavelet=wavelet,
                                      mode=dwt_mode)
                elif func == 'spect':
                    transf = spectrogram(x=data[feat].values,
                                         fs=spect_freq)[2][:,0]
                for i in range(0, len(transf)):
                    feat_grouped_i = feat_grouped + '_' + str(i)
                    data_grouped[feat_grouped_i] = transf[i]
                    feats_grouped.append(feat_grouped_i)
            elif func in ['min', 'max', 'avg', 'std', 'mode', 'median',
                          'sum', 'kurt']:
                if func == 'min':
                    data_grouped[feat_grouped] = data[feat].min()
                elif func == 'max':
                    data_grouped[feat_grouped] = data[feat].max()
                elif func == 'avg':
                    data_grouped[feat_grouped] = data[feat].mean()
                elif func == 'std':
                    data_grouped[feat_grouped] = data[feat].std()
                elif func == 'mode':
                    data_grouped[feat_grouped] = data[feat].mode()
                elif func == 'median':
                    data_grouped[feat_grouped] = data[feat].median()
                elif func == 'kurt':
                    data_grouped[feat_grouped] = data[feat].kurt()
                elif func == 'sum':
                    data_grouped[feat_grouped] = data[feat].sum()
                feats_grouped.append(feat_grouped)
            else:
                raise ValueError('{} is not a supported function.' \
                                 .format(func))
    for const in consts:
        data_grouped[const] = data[const].iloc[0]
        feats_grouped.append(const)
    return pd.Series(data_grouped, index=feats_grouped)

def GroupBy(data, columns, feats, consts, funcs=['min', 'max', 'avg', 'std',
                                                 'median', 'kurt'],
            fft_n=None, dwt_mode='symmetric', spect_freq=1):
    if type(data) not in [DataFrame, Series]:
        raise TypeError('\'data\' must be DataFrame or Series.')
    if type(columns) == list:
        for column in columns:
            if column not in data.columns:
                raise ValueError('{} is not a column of \'data\'.' \
                                 .format(column))
    else:
        raise TypeError('\'columns\' must be list.')
    if type(feats) != list:
        raise TypeError('\'feats\' must be list.')
    if type(funcs) != list:
        raise TypeError('\'funcs\' must be list.')
    if type(consts) != list and consts is not None:
        raise TypeError('\'consts\' must be list or NoneType.')
    if type(fft_n) != int and fft_n is not None:
        raise TypeError('\'fft_n\' must be integer or NoneType.')
    if type(dwt_mode) != str:
        raise TypeError('\'dwt_mode\' must be string.')
    if type(spect_freq) != int:
        raise TypeError('\'spect_freq\' must be string.')
    data = data.groupby(by=columns) \
               .apply(func=Group, feats=feats, funcs=funcs, consts=consts,
                      fft_n=fft_n, dwt_mode=dwt_mode, spect_freq=spect_freq)
    print('Data grouped succesfully.')
    return data

def Resample(data, freq, feats=None, consts=None, func='avg'):
    if feats == None or type(data) == Series:
        data1 = data.copy()
    else:
        data1 = data[feats].copy()
    resampler1 = data1.resample(rule=freq, axis='index')
    if func == 'min':
        data1 = resampler1.min()
    elif func == 'max':
        data1 = resampler1.max()
    elif func == 'avg':
        data1 = resampler1.mean()
    elif func == 'std':
        data1 = resampler1.std()
    elif func == 'sum':
        data1 = resampler1.std()
    else:
        raise ValueError('\'{}\' is not a valid function.'.format(func))
    if type(consts) == list:
        data2 = data[consts].copy()
        resampler2 = data2.resample(rule=freq, axis='index')
        data2 = resampler2.first()
        data = pd.concat(objs=[data1, data2], axis='columns')
    else:
        data = data1
    return data
