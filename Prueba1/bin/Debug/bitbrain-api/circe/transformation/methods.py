# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:00:00 2021

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries.
# _____________________________________________________________________________

import numpy as np
import mne

import pandas as pd
from pandas.core.frame import DataFrame, Series
from pandas.core.indexes.datetimes import DatetimeIndex

from scipy.stats import zscore

from circe.transformation.functions import GroupBy, Resample
from circe.transformation.biosignal_functions import PreprocessEEG, ProcessEEG
from circe.utilities.file_functions import LoadCSV, LoadJSON, SaveCSV
from circe.utilities.typing import class_type_check, strn, intn, floatn, \
                                   listn, dictn, fi, fin, ls, lsn, dlsn



# %% PYTHON CLASSES FOR TRANSFORMATION
# We define several Python structures for transformation tasks.
# _____________________________________________________________________________

class DataContext:

    def __init__(self, data, sep=',', dec='.', n_rows=None, columns=None,
                 header=None, index=None, time_unit=None, time_format=None,
                 feats=None, normalize=False, weights=None, get_report=False,
                 chs=None, ch_bads=None, chunk=2, amp_filter=150e-6,
                 freq_filter='firwin2', target_values=None, target_dict=None,
                 eeg_process=False, metadata=None, scaling=10e-6,
                 annotations=None, plot_raw=False, plot_psd=False,
                 plot_epochs=False):
        self.__init = True
        self.__use_norm = False
        if type(eeg_process) != bool:
            raise TypeError('\'eeg_process\' must be boolean.')
        if type(data) == str:
            self.load_data(file=data, sep=sep, dec=dec, n_rows=n_rows,
                           columns=columns, header=header, chs=chs,
                           ch_bads=ch_bads, chunk=chunk, amp_filter=amp_filter,
                           freq_filter=freq_filter,
                           target_values=target_values,
                           target_dict=target_dict, eeg_process=eeg_process,
                           metadata=metadata, scaling=scaling,
                           annotations=annotations, plot_raw=plot_raw,
                           plot_psd=plot_psd, plot_epochs=plot_epochs)
        elif type(data) in [DataFrame, Series]:
            self.__eeg_process = eeg_process
            if eeg_process == True:
                raw = PreprocessEEG(data=data, metadata=metadata,
                                    scaling=scaling, annotations=annotations)
                self.__data = ProcessEEG(raw=raw, chs=chs, chunk=chunk,
                                         amp_filter=amp_filter,
                                         freq_filter=freq_filter,
                                         target_values=target_values,
                                         target_dict=target_dict,
                                         plot_raw=plot_raw, plot_psd=plot_psd,
                                         plot_epochs=plot_epochs)
            else:
                self.__data = data
        else:
            '\'data\' must initialize from string (path), DataFrame or Series.'
        if index is not None \
        or time_unit is not None \
        or time_format is not None:
            self.set_index(index=index, time_unit=time_unit,
                           time_format=time_format)
        else:
            self.freq = None
        self.feats = feats
        self.weights = weights
        if normalize == True:
            self.normalize(weights=self.weights)
        elif normalize == False:
            self.__data_norm = None
            self.__use_norm = False
        else:
            raise TypeError('\'normalize\' must be boolean.')
        if get_report == True:
            self.get_report()
        elif get_report != False:
            raise TypeError('\'quality\' must be boolean.')
        self.__init = False

    @property
    def data(self):
        if self.use_norm == False:
            data0 = self.__data
        else:
            data0 = self.__data_norm
        return data0

    @data.setter
    def data(self, data):
        if type(data) in [DataFrame, Series]:
            if self.use_norm == True:
                print('Setting modified data as .data value. Setting '
                      '.data_norm = None and .use_norm = False...')
            self.__data = data
            self.__data_norm = None
            self.__use_norm = False
        else:
            raise TypeError('\'data\' must be DataFrame or Series. To load '
                            'data from path, use the .load() method.')

    @property
    def feats(self):
        return self.__feats

    @feats.setter
    def feats(self, feats):
        if type(feats) not in [list, str] and feats is not None:
            raise TypeError('\'feats\' must be list, string or NoneType.')
        if type(self.__data) == DataFrame:
            if type(feats) == list:
                self.__feats = feats
            elif type(feats) == str:
                self.__feats = [feats]
            elif feats == None:
                self.__feats = self.__data.columns.tolist()
            for feat in self.__feats:
                if feat not in self.__data.columns:
                    print('Ignoring not included feature \'{}\'...' \
                          .format(feat))
                    self.__feats.remove(feat)
        elif type(self.__data) == Series:
            self.__feats = [self.__data.name]
        self.__feats_numeric = self.dtype_feats(dtype='numeric')
        self.__feats_time = self.dtype_feats(dtype='time')
        self.__feats_nominal = self.dtype_feats(dtype='nominal')

    @property
    def feats_numeric(self):
        return self.__feats_numeric

    @property
    def feats_time(self):
        return self.__feats_time

    @property
    def feats_nominal(self):
        return self.__feats_nominal

    @property
    def weights(self):
       return self.__weights

    @weights.setter
    def weights(self, weights):
        if type(weights) == dict:
            for key in weights.keys():
                if key not in self.feats:
                    raise ValueError('{} is not a feature.'.format(key))
                elif key not in self.feats_numeric:
                    raise ValueError('{} is not a numeric feature.' \
                                     .format(key))
                if weights[key] not in [float, int]:
                    raise ValueError('\'weights\' values must be numeric.')
        elif weights is not None:
            raise TypeError('\'weights\' must be dictionary.')
        self.__weights = weights

    @property
    def data_norm(self):
        return self.__data_norm

    @property
    def use_norm(self):
        return self.__use_norm

    @use_norm.setter
    def use_norm(self, use_norm):
        if type(use_norm) == bool:
            self.__use_norm = use_norm
        else:
            raise TypeError('\'use_norm\' must be boolean.')

    @property
    def eeg_process(self):
        return self.__eeg_process

    def load_data(self, file, sep=',', dec='.', n_rows=None, columns=None,
                  header=None, chs=None, ch_bads=None, chunk=2,
                  amp_filter=150e-6, freq_filter='firwin2', target_values=None,
                  target_dict=None, eeg_process=False, metadata=None,
                  scaling=10e-6, annotations=None, plot_raw=False,
                  plot_psd=False, plot_epochs=False):
        extension = file.split('.')[-1]
        if extension == 'csv':
            data0 = LoadCSV(path=file, sep=sep, dec=dec, n_rows=n_rows,
                            columns=columns, header=header)
            if eeg_process == True:
                self.__eeg_process = True
                raw = PreprocessEEG(data=data0, metadata=metadata,
                                    scaling=scaling, annotations=annotations)
                self.__data = ProcessEEG(raw=raw, chs=chs, ch_bads=ch_bads,
                                         chunk=chunk, amp_filter=amp_filter,
                                         freq_filter=freq_filter,
                                         target_values=target_values,
                                         target_dict=target_dict,
                                         plot_raw=plot_raw, plot_psd=plot_psd,
                                         plot_epochs=plot_epochs)
            else:
                self.__eeg_process = False
                self.__data = data0
        elif extension == 'json':
            self.__eeg_process = False
            self.__data = LoadJSON(path=file)
        elif extension == 'vhdr':
            self.__eeg_process = True
            raw = mne.io.read_raw(fname=file, preload=True)
            self.__data = ProcessEEG(raw=raw, chs=chs, ch_bads=ch_bads,
                                     chunk=chunk, amp_filter=amp_filter,
                                     freq_filter=freq_filter,
                                     target_values=target_values,
                                     target_dict=target_dict,
                                     plot_raw=plot_raw, plot_psd=plot_psd,
                                     plot_epochs=plot_epochs)
        return self

    def save_data(self, file, sep=',', dec='.', index=False):
        SaveCSV(path=file, data=self.data, sep=sep, dec=dec, index=index)
        return self

    def set_index(self,
                  index = None,
                  time_unit: strn = None,
                  time_origin: str = 'unix',
                  time_format: strn = None):
        if index is None and time_unit is None and time_format is None:
            raise TypeError('\'index\', \'time_unit\' or \'time_format\' must '
                            'be specified.')
        if type(index) in [list, str]:
            if self.__init == True:
                self.__data = self.data.set_index(keys=index)
            else:
                self.data = self.data.set_index(keys=index)
        if type(time_unit) == str or type(time_format) == str:
            if type(time_unit) == str and type(time_format) == str:
                raise TypeError('\'time_unit\' and \'time_format\' can\'t be '
                                'specified simultaneously.')
            if type(time_unit) == str:
                index0 = pd.to_datetime(arg=self.__data.index, unit=time_unit,
                                        origin=time_origin, utc=True)
            elif type(time_format) == str:
                index0 = pd.to_datetime(arg=self.__data.index,
                                        format=time_format, utc=True)
            if self.__init == True:
                self.__data.index = index0
            else:
                self.data.index = index0
            self.freq = pd.infer_freq(self.__data.index)
        else:
            self.freq = None
        return self

    def dtype_feats(self, dtype):
        if type(self.data) == DataFrame:
            types = {'numeric': 'number',
                     'time': ['datetime', 'timedelta'],
                     'nominal': ['category', 'object']}
            feats0 = list(self.__data[self.feats]\
                              .select_dtypes(include=types[dtype]) \
                              .columns)
        if type(self.data) == Series:
            types = {'numeric': [float, int, np.float64, np.int64],
                     'time': [np.datetime64],
                     'nominal': [str]}
            if self.data.dtype in types[dtype]:
                feats0 = [self.data.name]
            else:
                feats0 = []
        return feats0

    def normalize(self, feats=None, weights=None, data_norm_path=None,
                  use_norm=True):
        data0 = self.data.copy()
        if type(feats) == list:
            feats0 = [feat for feat in feats \
                      if feat in self.feats_numeric]
        elif feats is None:
            feats0 = self.feats_numeric
        else:
            raise TypeError('\'feats\' must be list or NoneType.')
        for feat in feats0:
            if data0[feat].std() == 0:
               data0[feat] = 0
            else:
                data0[feat] = zscore(data0[feat].values)
        self.__data_norm = data0
        print('{} features normalized.'.format(len(feats0)))
        if type(weights) == dict:
            for feat in feats0:
                data0[feat] = data0[feat] * weights[feat]
            self.__data_norm = data0
            print('{} features weighted.'.format(len(feats0)))
        elif weights is not None:
            raise TypeError('\weights\' must be dict or NoneType.')
        if type(data_norm_path) == str:
            SaveCSV(path=data_norm_path, data=data0)
            print('Normalized data saved to \'{}\'.'.format(data_norm_path))
        elif data_norm_path is not None:
            raise TypeError('\'data_norm_path\' must be string or NoneType.')
        self.use_norm = use_norm
        return self

    def get_report(self, dimensions=['na', 'completeness', 'consistency',
                                     'outliers_5s', 'outliers_7s'],
                   report_path=None):
        if type(dimensions) != list:
            raise TypeError('\dimensions\' must be list.')
        report = pd.DataFrame(index=self.feats)
        for feat in self.feats:
            if feat in self.feats_numeric:
                report.loc[feat, 'Type'] = 'numeric'
            else:
                report.loc[feat, 'Type'] = 'nominal'
            if 'na' in dimensions:
                report.loc[feat, 'NA'] = len(self.data[self.data[feat].isna()])
            if 'not_na' in dimensions:
                report.loc[feat, 'Not NA'] \
                = len(self.data[self.data[feat].notna()])
            if 'completeness' in dimensions:
                report.loc[feat, 'Completeness'] = self.completeness(feat=feat)
            if 'consistency' in dimensions:
                report.loc[feat, 'Consistency'] = self.consistency(feat=feat)
            if 'outliers_3s' in dimensions:
                report.loc[feat, 'Outliers 3-sigma'] \
                = len(self.data[self.outliers(feat=feat, m=3)])
            if 'outliers_5s' in dimensions:
                report.loc[feat, 'Outliers 5-sigma'] \
                = len(self.data[self.outliers(feat=feat, m=5)])
            if 'outliers_7s' in dimensions:
                report.loc[feat, 'Outliers 7-sigma'] \
                = len(self.data[self.outliers(feat=feat, m=7)])
        report.loc['GLOBAL', 'Type'] \
        = '{}/{}'.format(len(report[report['Type']=='numeric']),
                         len(self.feat))
        if 'na' in dimensions:
            report.loc['GLOBAL', 'NA'] = report['NA'].sum()
        if 'not_na' in dimensions:
            report.loc['GLOBAL', 'Not NA'] = report['Not NA'].sum()
        if 'completeness' in dimensions:
            report.loc['GLOBAL', 'Completeness'] \
            = round(report['Completeness'].mean())
        if 'consistency' in dimensions:
            report.loc['GLOBAL', 'Consistency'] \
            = '{}/{}'.format(len(report[report['Consistency']==True]),
                             len(self.feats))
        if 'outliers_3s' in dimensions:
            report.loc['GLOBAL', 'Outliers 3-sigma'] \
            = report['Outliers 3-sigma'].sum()
        if 'outliers_5s' in dimensions:
            report.loc['GLOBAL', 'Outliers 5-sigma'] \
            = report['Outliers 5-sigma'].sum()
        if 'outliers_7s' in dimensions:
            report.loc['GLOBAL', 'Outliers 7-sigma'] \
            = report['Outliers 7-sigma'].sum()
        self.report = report
        print('Quality report generated.')
        if type(report_path) == str:
            SaveCSV(path=report_path, data=report)
            print('Quality report saved to \'{}\'.'.format(report_path))
        return self

    def completeness(self, feat):
        if type(feat) != str:
            raise TypeError('\'feat\' must be string.')
        if feat in self.feats:
            N = len(self.data)
            not_na = len(self.data[self.data[feat].notna()])
            completeness = round((not_na) * 100 / N)
        else:
            raise ValueError('{} is not in \'feats\'.'.format(feat))
        return completeness

    def consistency(self, feat):
        if type(feat) != str:
            raise TypeError('\'feat\' must be string.')
        if feat in self.feats_numeric:
            N = len(self.data)
            std = self.data[feat].std()
            error = std / (N ** 0.5)
            if error <= 2:
                consistency = True
            else:
                consistency = False
        elif feat in self.feats:
            values = self.data[feat].nunique()
            if values <= 100:
                consistency = True
            else:
                consistency = False
        else:
            raise ValueError('{} is not in \'feats\'.'.format(feat))
        return consistency

    def outliers(self, feat, m):
        if type(feat) != str:
            raise TypeError('\'feat\' must be string.')
        if type(m) != float:
            raise TypeError('\'m\' must be float.')
        if m <= 0:
            raise ValueError('\'m\' must be positive.')
        if feat in self.feats_numeric:
            mean = self.data[feat].mean()
            std = self.data[feat].std()
            outliers = (self.data[feat] < mean - std * m) \
                       |(self.data[feat] > mean + std * m)
        elif feat in self.feats:
            outliers = self.data[feat] == False
        else:
            raise ValueError('{} is not in \'feats\'.'.format(feat))
        return outliers

    def remove_missing(self, feats=None, remove_feats=None):
        if type(feats) == list:
            feats0 = [feat for feat in feats if feat in self.feats]
        elif feats is None:
            feats0 = self.feats
        else:
            raise TypeError('\feats\' must be list or NoneType.')
        original_size = len(self.data)
        if type(remove_feats) == float:
            feats1 = feats0[self.data.isna().mean() < remove_feats]
            self.data = self.data[feats1]
            self.feats = feats1
        elif remove_feats is not None:
            raise TypeError('\'remove_feats\' must be float or NoneType.')
        self.data = self.data.dropna(axis=0, subset=feats0, how='any')
        new_size = len(self.data)
        removed_frac = (original_size - new_size) / original_size
        print('{:.2%} of the rows removed from the data.'
              .format(removed_frac))
        return self

    def replace_missing(self, value=None, rolling_window=3, feats=None):
        if type(feats) == list:
            feats0 = [feat for feat in feats if feat in self.feats]
        elif feats is None:
            feats0 = self.feats
        else:
            raise TypeError('\feats\' must be list or NoneType.')
        for feat in feats0:
            if bool(value) == True:
                value = value
            elif bool(rolling_window) == True:
                value = self.data[feat].rolling(window=rolling_window) \
                                       .mean()
            else:
                value = self.data[feat].mean()
            self.data[self.data[feat].isna(), feat] = value
        print('Missing values replaced for {} features.' \
              .format(len(feats0)))
        return self

    def replace_outliers(self, m, value=None, rolling_window=3, feats=None):
        if type(feats) == list:
            feats0 = [feat for feat in feats if feat in self.feats_numeric]
        elif feats is None:
            feats0 = self.feats_numeric
        else:
            raise TypeError('\feats\' must be list or NoneType.')
        for feat in feats0:
            if bool(value) == True:
                value = value
            if bool(rolling_window) == True:
                value = self.data[feat].rolling(window=rolling_window) \
                                       .mean()
            else:
                value = self.data[feat].mean()
            self.data[self.outliers(feat=feat, m=m), feat] = value
        print('Outliers replaced for {} features.'.format(len(feats0)))
        return self

    def complete_time(self, freq=None):
        if type(self.data.index) != DatetimeIndex:
            raise TypeError('Index is not in the supported format.')
        if type(freq) == str:
            freq0 = freq
        else:
            freq0 = self.data.index.inferred_freq
        time_index = pd.date_range(start=self.data.index.min(),
                                   end=self.data.index.max(),
                                   freq=freq0)
        self.data = self.data.reindex(index=time_index)
        self.freq = freq0
        print('Time index completed.')
        return self

    def extra_feats(self, funcs=['cos', 'sin', 'tan'], feats=None):
        if type(funcs) != list:
            raise TypeError('\'funcs\' must be list.')
        if type(feats) == list:
            feats0 = [feat for feat in feats if feat in self.feats_numeric]
        elif feats is None:
            feats0 = self.feats_numeric
        else:
            raise TypeError('\feats\' must be list or NoneType.')
        for feat in feats0:
            for func in funcs:
                feat_extra = feat + '_' + func.upper()
                if func == 'cos':
                    self.data[feat_extra] = np.cos(self.data[feat])
                elif func == 'sin':
                    self.data[feat_extra] = np.sin(self.data[feat])
                elif func == 'tan':
                    self.data[feat_extra] = np.tan(self.data[feat])
                elif func == 'arctan':
                    self.data[feat_extra] = np.arctan(self.data[feat])
                elif func == '**2':
                    self.data[feat_extra] = self.data[feat]**2
                elif func == '**3':
                    self.data[feat_extra] = self.data[feat]**3
                else:
                    raise ValueError('{} is not a supported function.' \
                                     .format(func))
                self.feats.append(feat_extra)
        return self

    def group_by(self, columns, feats=None, consts=None, funcs=['min', 'max',
                                                                'avg', 'std',
                                                                'median',
                                                                'kurt'],
                 fft_n=None, dwt_mode='symmetric'):
        if type(consts) == list:
            consts0 = consts
        else:
            consts0 = []
        if type(feats) == list:
            feats0 = [feat for feat in feats if feat in self.feats_numeric \
                                             and feat not in consts]
        else:
            feats0 = self.feats_numeric.copy()
        self.data = GroupBy(data=self.data, columns=columns, feats=feats0,
                            funcs=funcs, consts=consts0, fft_n=fft_n,
                            dwt_mode=dwt_mode)
        self.feats = list(self.data.columns)
        return self

    def resample(self, freq, feats=None, consts=None, func='avg'):
        if type(feats) == list and type(self.data) == DataFrame:
            feats0 = [feat for feat in feats if feat in self.feats_numeric]
        elif type(self.data) == DataFrame:
            feats0 = self.feats_numeric.copy()
        else:
            feats0 = None
        self.data = Resample(data=self.data, freq=freq, feats=feats0,
                             consts=consts, func=func)
        self.freq = freq
        return self

    def append(self, other):
        self.data = self.data \
                        .append(other=other.data)
        if other.eeg_process == True:
            self.__eeg_process = True
        self.__feats = other.feats
        return self

    def drop_columns(self, columns):
        self.data = self.data \
                        .drop(labels=columns, axis='columns')
        self.feats = [feat for feat in self.feats if feat not in columns]
        print('{} columns droped.'.format(len(columns)))
        return self
