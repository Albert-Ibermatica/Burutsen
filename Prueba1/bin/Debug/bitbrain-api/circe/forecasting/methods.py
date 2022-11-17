# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:15:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries and modules.
# _____________________________________________________________________________

import warnings
import re
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from numpy import polyfit, sqrt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

from tqdm import trange

from circe.transformation.methods import DataContext
from circe.forecasting.parameters import algo_param, param_deflt, \
                                         algo_forbid
from circe.utilities.file_functions import CheckPath, SaveCSV, SavePKL, LoadPKL
from circe.utilities.typing import class_type_check, strn, intn, floatn, \
                                   listn, dictn, fi, fin, ls, lsn, dlsn


# %% WARNINGS
# We disable the warnings messages for a cleaner output.
# _____________________________________________________________________________

warnings.filterwarnings('ignore')



# %% PYTHON CLASSES FOR FORECASTING
# We define several Python structures for time-series analysis.
# _____________________________________________________________________________

class ForeContext(DataContext):

    def __init__(self,
                 data,
                 sep: str = ',',
                 dec: str = '.',
                 n_rows: intn = None,
                 columns = None,
                 header: intn = None,
                 values: strn = None,
                 index: strn = None,
                 time_unit: strn = None,
                 time_format: strn = None,
                 train_frac: floatn = 0.6):
        if type(data) == DataContext:
            self.__data = data.__data
            self.freq = data.freq
            self.__data_norm = None
            self.use_norm = False
        else:
            DataContext.__init__(self, data=data, sep=sep, dec=dec,
                                 n_rows=n_rows, columns=None, header=header,
                                 index=index, time_unit=time_unit,
                                 time_format=time_format, feats=[values],
                                 normalize=False, weights=None)
        self.__data_size = len(self.data)
        self.train_frac = train_frac
        if type(self.data.index) != DatetimeIndex:
            raise Exception('Index must be DatetimeIndex.')
        if type(self.data) == DataFrame and values is None:
            raise Exception('\'values\' must be specified when \'data\' is '
                            'DataFrame.')
        elif type(values) == str:
            if values in self.data.columns:
                self.data = self.data.loc[:, values]
            else:
                raise ValueError('\values\' must be a column.')
        self.feats = [self.data.name]
        self.__n_models = 0
        self.best_model = None

    @DataContext.data.setter
    def data(self, data):
        DataContext.data.fset(self, data)
        self.__data_size = len(data)
        self.__train_size = int(self.__data_size*self.__train_frac)
        self.__data_train = data.copy() \
                                .iloc[0:
                                      self.__train_size]
        self.__test_size = self.__data_size \
                         - self.__train_size
        self.__data_test = data.copy() \
                               .iloc[self.__train_size:
                                     self.__train_size+self.__test_size]

    @property
    def data_size(self):
        return self.__data_size

    @property
    def train_frac(self):
        return self.__train_frac

    @train_frac.setter
    def train_frac(self,
                   train_frac: float):
        if (0 <= train_frac) and (train_frac <= 1):
            self.__train_frac = train_frac
        else:
            raise ValueError('\'train_frac\' value must be between 0 and 1.')
        self.__train_size = int(self.__data_size*self.__train_frac)
        self.__data_train = self.data \
                                .iloc[0: self.__train_size] \
                                .copy()
        self.__test_size = self.__data_size \
                         - self.__train_size
        self.__data_test = self.data \
                               .iloc[self.__train_size:
                                     self.__train_size+self.__test_size] \
                               .copy()

    @property
    def train_size(self):
        return self.__train_size

    @property
    def data_train(self):
        return self.__data_train

    @property
    def test_size(self):
        return self.__test_size

    @property
    def data_test(self):
        return self.__data_test

    def save_data(self,
                  file: str,
                  sep: str = ',',
                  dec: str = '.',
                  dset: str = 'data'):
        if dset == 'data':
            SaveCSV(path=file, data=self.data, sep=sep, dec=dec, index=True)
        elif dset == 'train':
            SaveCSV(path=file, data=self.data_train, sep=sep, dec=dec,
                    index=True)
        elif dset == 'test':
            SaveCSV(path=file, data=self.data_test, sep=sep, dec=dec,
                    index=True)
        elif dset == 'pred':
            SaveCSV(path=file, data=self.data_pred, sep=sep, dec=dec,
                    index=True)
        return self

    def init_n_models(self,
                      model_param: dict):
        if type(model_param['algo']) is list:
            algo_values = model_param['algo']
        elif type(model_param['algo']) is str:
            algo_values = [model_param['algo']]
        for algo_value in algo_values:
            model_param_i = {}
            model_param_i['algo'] = algo_value
            param_list = [param for param in model_param \
                          if param in algo_param[algo_value]]
            for param in param_list:
                if type(model_param[param]) is list:
                    param_values = model_param[param]
                else:
                    param_values = [model_param[param]]
                for param_value in param_values:
                    if param not in model_param_i:
                        model_param_i[param] = param_value
                    else:
                        self.init_model_i(model_param_i=model_param_i,
                                          i=self.__n_models)
                        model_param_i[param] = param_value
            self.init_model_i(model_param_i=model_param_i, i=self.__n_models)
        print('{} model(s) initialized.'.format(self.__n_models))

    def init_model_i(self,
                     model_param_i: dict,
                     i: int):
        if self.check_param(model_param=model_param_i) == True:
            model_i = Model(freq=self.freq, model_param=model_param_i)
            model_i.long_name = 'm{}-{}'.format(i, model_i.name)
            setattr(self, '__{}'.format(model_i.long_name), model_i)
            self.__n_models += 1

    def check_param(self,
                    model_param: dict):
        flag = True
        if model_param['algo'] in algo_forbid:
            for combi in algo_forbid[model_param['algo']]:
                (m, n) = (0, 0)
                for param in combi:
                    if param in model_param:
                        if model_param[param] == combi[param]:
                            n += 1
                    elif param in param_deflt:
                        if param_deflt[param] == combi[param]:
                            n += 1
                    m += 1
                if m == n:
                    flag = flag and False
                else:
                    flag = flag and True
        return flag

    def choose(self,
               model_param: dict,
               n_best: intn = 3,
               plot_fore: bool = False):
        self.init_n_models(model_param=model_param)
        model_attr_names = [attr for attr in self.__dict__.keys() \
                            if attr.startswith('__m')]
        self.ranking = pd.DataFrame(columns=['algo', 'params', 'rmse'])
        self.inference = []
        for i in trange(len(model_attr_names), desc='Evaluating', ascii=" #",
                        unit='model'):
            model_attr_name = model_attr_names[i]
            model_i = getattr(self, model_attr_name)
            data_pred_i = model_i.fit(data_train=self.data_train) \
                                 .fore(horizon=self.test_size)
            model_i.rmse = model_i.score(data_test=self.data_test,
                                         data_pred=data_pred_i)
            rank_i = pd.Series({'name': model_i.name,
                                'long_name': model_i.long_name,
                                'algo': model_i.algo,
                                'params': model_i.__dict__,
                                'rmse': model_i.rmse},
                               name=model_i.long_name)
            self.ranking = self.ranking.append(rank_i)
        self.ranking = self.ranking.sort_values(by='rmse',
                                                ascending=True)
        if type(n_best) == int:
            self.ranking = self.ranking.iloc[0: n_best, :]
        self.best_model = getattr(self, '__'+self.ranking.index[0])
        self.data_pred = self.best_model.data_pred
        self.display()
        if plot_fore == True:
            self.plot_fore()
        return self

    def display(self):
        print('Displaying results of {} method(s)...' \
              .format(len(self.ranking)))
        rmse_str = str(round(self.best_model.rmse, 2))
        best_param = self.ranking.loc[self.ranking.index[0], 'params']
        param_str_list = ['.{} = {}'.format(param, best_param[param]) \
                          for param in best_param.keys() \
                          if param not in ['name', 'long_name', 'algo',
                                           'method', 'vector', 'rmse',
                                           'freq', 'data_train', 'location',
                                           'horizon','data_pred',
                                           'comp_seasonal', 'comp_trend',
                                           'comp_resid']] \
                       + ['.{}: {}'.format(param,
                                           getattr(self.best_model.method,
                                                   param).name) \
                          for param in ['model_seasonal', 'model_trend',
                                        'model_resid']]
        param_str = ' \n                  '.join(param_str_list)
        param_str_sizes = [len(param) + 18 for param in param_str_list]
        display1 = '  -Name:          {} \n'.format(self.best_model.name)
        display2 = '  -Algorithm:     {} \n'.format(self.best_model.algo)
        display3 = '  -Parameters:    {} \n'.format(param_str)
        display4 = '  -RMSE:          {} \n'.format(rmse_str)
        if len(self.ranking) > 1:
            alt_str_list = ['#{}: {} (RMSE: {})' \
                            .format(i+1,
                                    self.ranking \
                                        .loc[self.ranking.index[i], 'name'],
                                    round(self.ranking \
                                              .loc[self.ranking.index[i],
                                                   'rmse'],
                                          2)) \
                            for i in range(1, len(self.ranking))]
            alt_str = ' \n                  '.join(alt_str_list)
            alt_str_sizes = [len(alt) + 18 for alt in alt_str_list]
            display5 = '  -Alternatives:  {} \n'.format(alt_str)
        else:
            alt_str_sizes = []
            display5 = ''
        max_size = max([len(display1), len(display2), len(display4)] \
                       +param_str_sizes+alt_str_sizes)
        display0 = '-' * (max_size - 1)
        display = '\n' + display0 + '\n' + display1 + display2 + display3 \
                + display4 + display5
        return print(display)

    def plot_fore(self):
        fig = plt.figure()
        fig.set_size_inches(15, 10)
        ax = fig.add_subplot(len(self.ranking)+1, 1, 1)
        ax.plot(self.data)
        ax.set_title('Historical series', loc='right')
        time_format = mdates.ConciseDateFormatter(self.data.index)
        ax.xaxis.set_major_formatter(time_format)
        time_test_format = mdates.ConciseDateFormatter(self.data_test.index)
        for i in range(0, len(self.ranking)):
            ax = fig.add_subplot(len(self.ranking)+1, 1, i+2)
            model_i = getattr(self,
                              '__{}'.format(self.ranking \
                                                .loc[self.ranking.index[i],
                                                     'long_name']))
            ax.plot(self.data_test)
            ax.plot(model_i.data_pred)
            ax.set_title('{}'.format(model_i.long_name),
                         loc='right')
            ax.xaxis.set_major_formatter(time_test_format)
        fig.tight_layout()
        return self

    def fit_save(self,
                 model_param: dict,
                 folder: str):
        self.train_frac = 1.0
        setattr(self,
                '__model{}'.format(self.__n_models+1),
                Model(model_param=model_param) \
                .fit(data_train=self.data_train) \
                .save(folder=folder))
        return self

    def fit_best(self):
        self.train_frac = 1.0
        if self.best_model is None:
            raise Exception('Unable to apply \.fit_best\' when no model has '
                            'been choosen the best. Please apply \'.choose\' '
                            'method first.')
        self.best_model \
            .fit(data_train=self.data_train)
        return self

    def save_best(self,
                  folder: str):
        if self.best_model is None:
            raise Exception('Unable to apply \.save_best\' when no model has'
                            'been choosen the best. Please apply \'.choose\' '
                            'method first.')
        self.best_model \
            .save(folder=folder)
        return self

    def fit_save_best(self,
                      folder: str):
        self.train_frac = 1.0
        if self.best_model is None:
            raise Exception('Unable to apply \.fit_save_best\' when no model '
                            'has been choosen the best. Please apply '
                            '\'.choose\' method first.')
        self.best_model \
            .fit(data_train=self.data_train) \
            .save(folder=folder)
        return self

    def load_fore(self,
                  folder: str,
                  horizon: int):
        self.train_frac = 0.0
        CheckPath(path=folder, to_be='folder')
        model = LoadPKL(path=folder+'model.pkl')
        self.inference = [model.long_name]
        self.data_pred = model.fore(horizon=horizon)
        return self.data_pred

    def fit_fore_best(self,
                      horizon: int):
        self.train_frac = 1.0
        if self.best_model is None:
            raise Exception('Unable to apply \.fit_save_best\' when no model '
                            'has been choosen the best. Please apply '
                            '\'.choose\' method first.')
        self.inference = [self.best_model.long_name]
        self.data_pred = self.best_model \
                             .fit(data_train=self.data_train) \
                             .fore(horizon=horizon)
        return self.data_pred

class Model():

    def __init__(self,
                 freq: str,
                 model_param: dict):
        self.freq = freq
        if type(model_param) is dict:
            if 'algo' in model_param:
                self.algo = model_param['algo']
        elif type(model_param) is str:
            self.algo = model_param
        else:
            raise TypeError('\'model_param\' must be dictionary or string.')
        for param in algo_param[self.algo]:
            setattr(self, param, param_deflt[param])
            if type(model_param) is dict:
                if param in model_param:
                    if type(model_param[param]) in [dict, str, float, int,
                                                    bool] \
                    or model_param[param] is None:
                        setattr(self, param, model_param[param])
                    else:
                        raise TypeError('Parameters must be dict, string, '
                                        'float, integer, dictionary, boolean '
                                        'or NoneType.')
        if self.algo == 'poly':
            self.method = TimePoly(freq=self.freq, cycle=self.poly_cycle,
                                   d=self.poly_d)
        elif self.algo == 'comp':
            self.method = Composed(freq=self.freq,
                                   cycle=self.comp_cycle,
                                   rule=self.comp_rule,
                                   seasonal=self.comp_seasonal,
                                   trend=self.comp_trend,
                                   resid=self.comp_resid)
        self.get_name()

    def get_name(self):
        if self.algo == 'poly':
            self.name = 'p:{}'.format(self.poly_d)
        elif self.algo == 'arima':
            self.name = 'a:{}:{}:{}'.format(self.arima_p, self.arima_d,
                                           self.arima_q)
        elif self.algo == 'comp':
            seasonal_name = self.method.model_seasonal.name
            trend_name = self.method.model_trend.name
            resid_name = self.method.model_resid.name
            self.name = '{}-{}-{}'.format(seasonal_name, trend_name,
                                          resid_name)
        return self

    def fit(self, data_train):
        self.data_train = data_train
        if self.algo == 'poly':
            self.vector = self.method\
                              .fit(data_train=self.data_train)
        elif self.algo == 'arima':
            self.location = len(self.data_train)
            self.vector = ARIMA(endog=self.data_train,
                                order=(self.arima_p, self.arima_d,
                                       self.arima_q)) \
                          .fit()
        elif self.algo == 'comp':
            self.vector = self.method.fit(data_train=self.data_train)
        return self

    def score(self, data_test, data_pred):
        rmse = sqrt(mean_squared_error(y_true=data_test, y_pred=data_pred))
        return rmse

    def fore(self, horizon):
        self.horizon = horizon
        if self.algo == 'poly':
            self.data_pred = self.vector.fore(horizon=self.horizon)
        elif self.algo == 'arima':
            self.data_pred \
            = self.vector.predict(start=self.location,
                                  end=self.location+self.horizon-1)
        elif self.algo == 'comp':
            self.data_pred = self.vector.fore(horizon=self.horizon)
        return self.data_pred

    def save(self,
             folder: str):
        CheckPath(path=folder, to_be='folder')
        SavePKL(obj=self, path=folder+'model.pkl')
        return self

class TimePoly():

    def __init__(self,
                 freq: str,
                 cycle: str,
                 d: int):
        self.freq = freq
        self.cycle = cycle
        self.d = d

    def fit(self, data_train):
        self.data_train = data_train
        self.time_train = self.data_train.index
        self.time_train_poly = self.get_time_poly(time=self.time_train)
        self.vector = polyfit(x=self.time_train_poly,
                              y=self.data_train,
                              deg=self.d)
        return self

    def get_time_poly(self, time):
        if self.cycle is None:
            poly_time = [t for t in range(1, len(time)+1)]
        elif self.cycle == 'daily':
            poly_time = time.hour
        elif self.cycle == 'weekly':
            poly_time= time.dayofweek
        return poly_time

    def fore(self, horizon):
        self.horizon = horizon
        self.time_pred = self.get_time_pred(time=self.time_train)
        self.time_pred_poly = self.get_time_poly(time=self.time_pred)
        self.data_pred = pd.Series(dtype='float64')
        for t in range(self.horizon):
            self.data_pred[self.time_pred[t]] = self.vector[-1]
            for i in range(self.d):
                self.data_pred[self.time_pred[t]] \
                += self.time_pred_poly[t]**(self.d-i) \
                 * self.vector[i]
        return self.data_pred

    def get_time_pred(self, time):
        freq_regex = re.search('(^\d*)(D|H|T{1}$)', self.freq)
        if freq_regex[1] == '':
            c1 = 1
        else:
            c1 = int(freq_regex[1])
        if freq_regex[2] == 'H':
            c2 = 1
        elif freq_regex[2] == 'T':
            c2 = 60
        elif freq_regex[2] == 'D':
            c2 = 1/24
        time_pred_min = time.max() \
                      + dt.timedelta(hours=c1/c2)
        time_pred_max = time.max() \
                      + dt.timedelta(hours=c1/c2) * self.horizon
        time_pred = pd.date_range(start=time_pred_min, end=time_pred_max,
                                  freq=self.freq)
        return time_pred

class Composed():

    def __init__(self,
                 freq: str,
                 cycle: str,
                 rule: str,
                 seasonal: dict,
                 trend: dict,
                 resid: dict):
        self.freq = freq
        self.cycle = cycle
        self.rule = rule
        self.model_seasonal = Model(freq=self.freq, model_param=seasonal)
        self.model_trend = Model(freq=self.freq, model_param=trend)
        self.model_resid = Model(freq=self.freq, model_param=resid)

    def fit(self, data_train):
        self.period = self.get_period(data=data_train)
        data_comps = STL(endog=data_train, period=self.period).fit()
        self.model_seasonal.fit(data_train=data_comps.seasonal)
        self.model_trend.fit(data_train=data_comps.trend)
        self.model_resid.fit(data_train=data_comps.resid)
        return self

    def get_period(self, data):
        time_min = data.index.min()
        if self.cycle == 'daily':
            time_max = time_min + dt.timedelta(days=1)
        elif self.cycle == 'weekly':
            time_max = time_min + dt.timedelta(days=7)
        time_period = pd.date_range(start=time_min, end=time_max,
                                    freq=self.freq)
        period = len(time_period) - 1
        return period

    def save(self,
             folder: str):
        CheckPath(path=folder, to_be='folder')
        SavePKL(obj=self, path=folder+'model.pkl')
        return self

    def fore(self, horizon):
        self.horizon = horizon
        if self.rule == 'add':
            data_pred = self.model_seasonal.fore(horizon=self.horizon) \
                      + self.model_trend.fore(horizon=self.horizon) \
                      + self.model_resid.fore(horizon=self.horizon)
        elif self.rule == 'mult':
            data_pred = self.model_seasonal.fore(horizon=self.horizon) \
                      * self.model_trend.fore(horizon=self.horizon) \
                      * self.model_resid.fore(horizon=self.horizon)
        return data_pred
