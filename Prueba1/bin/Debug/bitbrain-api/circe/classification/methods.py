# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:00:00 2021

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries and modules.
# _____________________________________________________________________________

import warnings
import re
import pandas as pd

from statistics import mean

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import Sequential
from tensorflow.keras.layers import DenseFeatures, Dense, Dropout
from tensorflow.keras.models import load_model

from tqdm import trange

from circe.transformation.methods import DataContext
from circe.classification.parameters import algo_param, param_deflt, \
                                            algo_forbid, neural_algos
from circe.classification.functions import SelectKFeats, DataSet
from circe.utilities.file_functions import CheckPath, SavePKL, LoadPKL
from circe.utilities.typing import class_type_check, strn, intn, floatn, \
                                   listn, dictn, fi, fin, ls, lsn, dlsn



# %% WARNINGS
# We disable the warnings messages for a cleaner output.
# _____________________________________________________________________________

warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# %% PYTHON CLASSES FOR CLASSIFICATON
# We define several Python structures for classification tasks.
# _____________________________________________________________________________

# @class_type_check
class ClassContext(DataContext):

    def __init__(self,
                 data,
                 sep: str = ',',
                 dec: str = '.',
                 n_rows: intn = None,
                 columns = None,
                 header: intn = None,
                 index: lsn = None,
                 time_unit: strn = None,
                 time_format: strn = None,
                 feats: listn = None,
                 target: strn = None,
                 train_frac: floatn = 0.6,
                 test_frac: floatn = 1.0,
                 encode_target: bool = False,
                 select_k_feats: intn = None,
                 select_func: str = 'f_class',
                 chs: listn = None,
                 ch_bads: listn = None,
                 chunk: fi = 2,
                 amp_filter: floatn = 150e-6,
                 freq_filter: strn = 'firwin2',
                 target_values: listn = None,
                 target_dict: dictn = None,
                 eeg_process: bool = False,
                 metadata = None,
                 scaling: float = 10e-6,
                 annotations:dictn = None,
                 plot_raw: bool = False,
                 plot_psd:bool = False,
                 plot_epochs:bool = False):
        if type(data) == DataContext:
            self.__data = data.__data
            self.freq = data.freq
            self.feats = data.feats
            self.__data_norm = data.__data_norm
            self.use_norm = data.use_norm
        else:
            DataContext.__init__(self, data=data, sep=sep, dec=dec,
                                 n_rows=n_rows, columns=columns, header=None,
                                 index=index, time_unit=time_unit,
                                 time_format=time_format, feats=feats,
                                 normalize=False, weights=None, chs=chs,
                                 ch_bads=ch_bads, chunk=chunk,
                                 amp_filter=amp_filter,
                                 freq_filter=freq_filter,
                                 target_values=target_values,
                                 target_dict=target_dict,
                                 eeg_process=eeg_process, metadata=metadata,
                                 scaling=scaling, annotations=annotations,
                                 plot_raw=plot_raw, plot_psd=plot_psd,
                                 plot_epochs=plot_epochs)
        self.__data_size = len(self.data)
        self.train_frac = train_frac
        self.test_frac = test_frac
        if eeg_process == True:
            if type(target) == str:
                print('EEG data loaded: ignoring \'target\' value...')
            if annotations is not None:
                print('\'annotations\' passed: setting \'target\' value to ' \
                      'Target\'.')
                self.target = 'Target'
            else:
                print('\'annotations\' is None: setting \'target\' value to ' \
                      'None.')
                self.target = None
        else:
            self.target = target
        if encode_target == True:
            self.encode_target()
        else:
            self.target_decoder = None
        if type(select_k_feats) == int:
            self.select_k_feats(k=select_k_feats, select_func=select_func)
        self.__n_models = 0
        self.best_model = None

    @DataContext.data.setter
    def data(self, data):
        DataContext.data.fset(self, data)
        self.__data_size = len(data)
        self.__train_size = int(self.__data_size*self.__train_frac)
        self.__data_train = data.copy() \
                                .iloc[0:
                                      self.__train_size,
                                      :]
        self.__valid_size = int((1-self.__test_frac) \
                                *(self.__data_size-self.__train_size))
        self.__test_size = self.__data_size \
                         - self.__train_size \
                         - self.__valid_size
        self.__data_test = data.copy() \
                               .iloc[self.__train_size:
                                     self.__train_size+self.__test_size,
                                     :]
        if self.__valid_size > 0:
            self.__data_valid = data.copy() \
                                    .iloc[self.__test_size:
                                          self.__test_size+self.__valid_size,
                                          :]

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
            raise ValueError('\'train_frac\' value must be between 0 and '
                                 '1.')
        self.__train_size = int(self.__data_size*self.__train_frac)
        self.__data_train = self.data \
                                .iloc[0: self.__train_size,
                                      :] \
                                .copy()

    @property
    def train_size(self):
        return self.__train_size

    @property
    def data_train(self):
        return self.__data_train

    @property
    def test_frac(self):
        return self.__test_frac

    @test_frac.setter
    def test_frac(self,
                  test_frac: float):
        if (0 <= test_frac) and (test_frac <= 1):
            self.__test_frac = test_frac
        else:
            raise ValueError('\'test_frac\' value must be greater than 0 and '
                             'greater than or equal to 1.')
        self.__valid_size = int((1-self.__test_frac) \
                                *(self.__data_size-self.__train_size))
        self.__test_size = self.__data_size \
                         - self.__train_size \
                         - self.__valid_size
        self.__data_test = self.data \
                               .iloc[self.__train_size:
                                     self.__train_size+self.__test_size,
                                     :] \
                               .copy()
        if self.__valid_size > 0:
            self.__data_valid = self.data \
                                    .iloc[self.__test_size:
                                          self.__test_size+self.__valid_size,
                                          :] \
                                    .copy()
        else:
            self.__data_valid = None

    @property
    def test_size(self):
        return self.__test_size

    @property
    def data_test(self):
        return self.__data_test

    @property
    def valid_size(self):
        return self.__valid_size

    @property
    def data_valid(self):
        return self.__data_valid

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self,
               target: strn):
        self.__target = target
        if self.__target in self.feats:
            self.feats.remove(self.__target)

    @DataContext.use_norm.setter
    def use_norm(self, use_norm):
        DataContext.use_norm.fset(self, use_norm)
        self.train_frac = self.train_frac
        self.test_frac = self.test_frac

    @property
    def n_models(self):
        return self.__n_models

    def group_by(self,
                 columns:list,
                 funcs: list = ['min', 'max', 'avg', 'std', 'median', 'kurt'],
                 feats: listn = None,
                 consts: listn = None,
                 fft_n: intn = None,
                 dwt_mode: list = 'symmetric'):
        if type(consts) == list:
            if type(self.target) == str and self.target not in consts:
                consts.append(self.target)
        elif type(self.target) == str:
            consts = [self.target]
        else:
            consts = []
        super().group_by(columns=columns, funcs=funcs, feats=feats,
                         consts=consts, fft_n=fft_n, dwt_mode=dwt_mode)
        self.feats = [feat for feat in self.data.columns \
                      if feat != self.target]
        return self

    def append(self, other):
        super().append(other=other)
        self.__target = other.target
        self.train_frac = other.train_frac
        self.test_frac = other.test_frac
        return self

    def encode_target(self):
        if self.target is None:
            raise Exception('Unable to decode None \'target\'.')
        i = 0
        self.target_decoder = {}
        for value in self.data[self.target].unique():
            self.data.loc[self.data[self.target]==value, self.target] = i
            self.target_decoder[i] = value
            print('Target value {} encoded as {}.'.format(value, i))
            i += 1
        return self

    def decode_target(self):
        self.data[self.target] = self.data[self.target] \
                                     .replace(to_replace=self.target_decoder)
        self.target_decoder = None
        return self

    def select_k_feats(self,
                       k: intn = None,
                       select_func: str = 'f_class',
                       load_folder: strn = None,
                       save_folder: strn = None):
        if type(k) == int:
            feats0 = SelectKFeats(data=self.data, feats=self.feats,
                                  target=self.target, k=k,
                                  select_func=select_func)
        elif type(load_folder) == str:
            CheckPath(path=load_folder, to_be='folder')
            feats0 = LoadPKL(path=load_folder+'selected_feats.pkl')
        else:
            raise Exception('Either \'k\' or \'load_folder\' must be '
                            'specified.')
        if type(save_folder) == str:
            CheckPath(path=save_folder, to_be='folder')
            SavePKL(obj=feats0, path=save_folder+'selected_feats.pkl')
        self.feats = feats0
        if type(self.target) == str:
            self.data = self.data[feats0+[self.target]]
        elif self.target is None:
            self.data = self.data[feats0]
        return self

    def test(self,
             model_param: dict,
             cross_validate: intn = None,
             metric: str = 'accuracy',
             epochs: int = 1):
        self.init_model_i(model_param=model_param,
                          i=self.__n_models)
        self.metric = metric
        if cross_validate == 0:
            raise ValueError('\'cross_validate\' must be integer greater than '
                             '0.')
        elif bool(cross_validate) == True:
            valid_fold = bool(self.valid_size)
            self.__model.scores = self.__model \
                                      .build() \
                                      .cross_validate(data=self.data,
                                                      k=cross_validate,
                                                      feats=self.feats,
                                                      target=self.target,
                                                      epochs=epochs,
                                                      valid_fold=valid_fold)
        else:
            self.__model.scores = [self.__model \
                                       .build()
                                       .fit(data_train=self.data_train,
                                            feats=self.feats,
                                            target=self.target,
                                            data_valid=self.data_valid,
                                            epochs=epochs) \
                                       .score(data_test=self.data_test,
                                              feats=self.feats,
                                              target=self.target)]
        self.__model.scores_avg = mean(self.__model.scores)
        self.display(epochs=epochs)
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
            layer_str = '({})'.format(algo_value) + '(_layer)(\d+$)'
            param_list = [param for param in model_param \
                          if param in algo_param[algo_value] \
                          or re.search(layer_str, param) is not None]
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
            model_i = Model(model_param=model_param_i,
                            feats_neural=self.feats_numeric)
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
               cross_validate: intn = None,
               metric: str = 'accuracy',
               epochs: int = 1,
               n_best: intn = None,
               seed: int = 1):
        self.metric = metric
        self.init_n_models(model_param=model_param)
        model_attr_names = [attr for attr in self.__dict__.keys() \
                            if attr.startswith('__m')]
        self.ranking = pd.DataFrame(columns=['algo', 'params', 'scores',
                                    'scores_avg'])
        self.inference = []
        for i in trange(len(model_attr_names), desc='Evaluating', ascii=" #",
                        unit='model'):
            model_attr_name = model_attr_names[i]
            model_i = getattr(self, model_attr_name)
            self.inference.append(model_i.long_name)
            if cross_validate == 0:
                raise ValueError('\'cross_validate\' must be integer greater '
                                 'than 0.')
            elif bool(cross_validate) == True:
                valid_fold = bool(self.valid_size)
                model_i.scores = model_i.build() \
                                        .cross_validate(data=self.data,
                                                        k=cross_validate,
                                                        feats=self.feats,
                                                        target=self.target,
                                                        epochs=epochs,
                                                        valid_fold=valid_fold,
                                                        seed=seed)
            else:
                model_i.scores = [model_i.build() \
                                         .fit(data_train=self.data_train,
                                              feats=self.feats,
                                              target=self.target,
                                              data_valid=self.data_valid,
                                              epochs=epochs,
                                              seed=seed) \
                                         .score(data_test=self.data_test,
                                                feats=self.feats,
                                                target=self.target,
                                                seed=seed)]
            model_i.scores_avg = mean(model_i.scores)
            rank_i = pd.Series({'name': model_i.name,
                                'long_name': model_i.long_name,
                                'algo': model_i.algo,
                                'params': model_i.__dict__,
                                'scores': model_i.scores,
                                'scores_avg': model_i.scores_avg},
                               name=model_i.long_name)
            self.ranking = self.ranking.append(rank_i)
        self.ranking = self.ranking.sort_values(by='scores_avg',
                                                ascending=False)
        if type(n_best) == int:
            self.ranking = self.ranking.iloc[0: n_best, :]
        self.best_model = getattr(self, '__'+self.ranking.index[0])
        self.display(epochs=epochs)
        return self

    def display(self,
                epochs: int = 1):
        print('Displaying results of {} method(s)...' \
              .format(len(self.ranking)))
        scores_str_list = [str(round(score, 2)) for score in self.best_model \
                                                                 .scores]
        scores_str = ', '.join(scores_str_list)
        scores_avg_str = str(round(self.best_model.scores_avg, 2))
        best_param = self.ranking.loc[self.ranking.index[0], 'params']
        param_str_list = ['.{} = {}'.format(param, best_param[param]) \
                          for param in best_param.keys() \
                          if param not in ['name', 'long_name', 'algo',
                                           'method', 'vector', 'scores',
                                           'scores_avg', 'neural_inputs',
                                           'history'] \
                          and not param.startswith('layer')] \
                       + ['.{}: {}'.format(param, best_param[param].type) \
                          for param in best_param.keys() \
                          if param.startswith('layer')]
        param_str = ' \n                  '.join(param_str_list)
        param_str_sizes = [len(param) + 18 for param in param_str_list]
        if self.best_model.is_neural == True:
            display3 = '  -Layer number:  {} \n'.format(self.best_model \
                                                            .n_layers)
            display4 = '  -Epochs:        {} \n'.format(epochs)
        else:
            display3 = ''
            display4 = ''
        display1 = '  -Name:          {} \n'.format(self.best_model.name)
        display2 = '  -Algorithm:     {} \n'.format(self.best_model.algo)
        display5 = '  -Parameters:    {} \n'.format(param_str)
        display6 = '  -Metric:        {} \n'.format(self.metric)
        display7 = '  -Scores:        {} \n'.format(scores_str)
        display8 = '  -Average score: {} \n'.format(scores_avg_str)
        if len(self.ranking) > 1:
            alt_str_list = ['#{}: {} (average score: {})' \
                            .format(i+1,
                                    self.ranking.loc[self.ranking.index[i],
                                                     'name'],
                                    round(self.ranking.loc[self.ranking.index[i],
                                                           'scores_avg'],
                                          2)) \
                            for i in range(1, len(self.ranking))]
            alt_str = ' \n                  '.join(alt_str_list)
            alt_str_sizes = [len(alt) + 18 for alt in alt_str_list]
            display9 = '  -Alternatives:  {} \n'.format(alt_str)
        else:
            alt_str_sizes = []
            display9 = ''
        max_size = max([len(display1), len(display2), len(display3),
                        len(display4), len(display6), len(display7)] \
                        +param_str_sizes+alt_str_sizes)
        display0 = '-' * (max_size - 1)
        display = '\n' + display0 + '\n' + display1 + display2 + display3 \
                + display4 + display5 + display6 + display7 + display8 \
                + display9
        return print(display)

    def fit_save(self,
                 model_param: dict,
                 folder: str):
        self.train_frac = 1.0
        self.test_frac = 1.0
        setattr(self,
                '__model{}'.format(self.__n_models+1),
                Model(model_param=model_param) \
                .fit(data_train=self.data_train, feats=self.feats,
                     target=self.target) \
                .save(folder=folder))
        return self

    def fit_best(self):
        self.train_frac = 1.0
        self.test_frac = 1.0
        if self.best_model is None:
            raise Exception('Unable to apply \.fit_best\' when no model has '
                            'been choosen the best. Please apply \'.choose\' '
                            'method first.')
        self.best_model \
            .fit(data_train=self.data_train, feats=self.feats,
                 target=self.target)
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
        self.test_frac = 1.0
        if self.best_model is None:
            raise Exception('Unable to apply \.fit_save_best\' when no model '
                            'has been choosen the best. Please apply '
                            '\'.choose\' method first.')
        self.best_model \
            .fit(data_train=self.data_train, feats=self.feats,
                 target=self.target) \
            .save(folder=folder)
        return self

    def load_apply(self,
                   folder: str):
        self.train_frac = 0.0
        self.test_frac = 1.0
        CheckPath(path=folder, to_be='folder')
        model = LoadPKL(path=folder+'model.pkl')
        if model.is_neural == True:
            model.method = load_model(folder+'neural_model')
        self.inference = [model.long_name]
        self.data_result = model.apply(data_test=self.data_test,
                                       feats=self.feats)
        return self.data_result

# @class_type_check
class Model():

    def __init__(self,
                 model_param: dict,
                 feats_neural: listn = None):
        if type(model_param) is dict:
            if 'algo' in model_param:
                self.algo = model_param['algo']
        elif type(model_param) is str:
            self.algo = model_param
        else:
            raise TypeError('\'model_param\' must be dictionary or string.')
        if self.algo in neural_algos:
            self.is_neural = True
            self.get_neural_inputs(feats_neural=feats_neural)
            self.n_layers = 0
            for param in model_param.keys():
                layer_str = '({})'.format(self.algo) + '(_layer)(\d+$)'
                layer_regex = re.search(layer_str, param)
                if layer_regex is not None:
                    if int(layer_regex[3]) == self.n_layers + 1:
                        setattr(self, 'layer{}' \
                                .format(self.n_layers+1),
                                Layer(layer_param=model_param[param],
                                      name=param))
                        self.n_layers += 1
                    else:
                        raise ValueError('Neural network layers must be '
                                         'included numbered and ordered. '
                                         'For example: '
                                         '\'seq_layer1\': {}, '
                                         '\'seq_layer2\': {}.')
            if self.n_layers == 0:
                raise ValueError('Neural networks must include at least one '
                                 'layer.')
        else:
            self.is_neural = False
            self.neural_inputs = None
            self.n_layers = None
        for param in algo_param[self.algo]:
            setattr(self, param, param_deflt[param])
            if type(model_param) is dict:
                if param in model_param:
                    if type(model_param[param]) in [str, float, int, bool] \
                    or model_param[param] is None:
                        setattr(self, param, model_param[param])
                    else:
                        raise TypeError('Parameters must be string, float, '
                                        'integer, dictionary, boolean or '
                                        'NoneType.')
        self.get_name()

    def get_name(self):
        if self.algo in ['linr', 'logr']:
            self.name = self.algo
        elif self.algo == 'svc':
            self.name = 'svc:{}'.format(self.svc_kernel)
        elif self.algo == 'tree':
            self.name = 'tree:{}'.format(self.tree_criterion)
        elif self.algo == 'forest':
            self.name = 'forest:{}:{}'.format(self.forest_n_estimators,
                                             self.forest_criterion)
        elif self.algo == 'lda':
            self.name = 'lda:{}:{}'.format(self.lda_solver,
                                          self.lda_n_components)
        elif self.algo == 'seq':
            self.name = 'seq:{}'.format(self.n_layers)
        return self

    def get_neural_inputs(self,
                          feats_neural: list):
        self.neural_inputs = []
        for feat_neural in feats_neural:
            self.neural_inputs \
                .append(feature_column.numeric_column(feat_neural))
        return self

    def build(self):
        if self.algo == 'linr':
            self.method = LinearRegression(n_jobs=self.linr_n_jobs)
        elif self.algo == 'logr':
            self.method \
            = LogisticRegression(C=self.logr_c,
                                 penalty=self.logr_penalty,
                                 class_weight=self.logr_class_weight,
                                 tol=self.logr_tol,
                                 max_iter=self.logr_max_iter,
                                 n_jobs=self.logr_n_jobs)
        elif self.algo == 'svc':
            self.method \
            = SVC(C=self.svc_c,
                  kernel=self.svc_kernel,
                  degree=self.svc_d,
                  gamma=self.svc_gamma,
                  class_weight=self.svc_class_weight,
                  tol=self.svc_tol,
                  max_iter=self.svc_max_iter)
        elif self.algo == 'tree':
            self.method \
            = DecisionTreeClassifier(criterion=self.tree_criterion,
                                     splitter=self.tree_splitter,
                                     max_depth=self.tree_max_depth,
                                     min_samples_split=self.tree_min_split,
                                     min_samples_leaf=self.tree_min_leaf,
                                     max_features=self.tree_max_features,
                                     class_weight=self.tree_class_weight)
        elif self.algo == 'forest':
            self.method \
            = RandomForestClassifier(n_estimators=self.forest_n_estimators,
                                     criterion=self.forest_criterion,
                                     max_depth=self.forest_max_depth,
                                     min_samples_split=self.forest_min_split,
                                     min_samples_leaf=self.forest_min_leaf,
                                     max_features=self.forest_max_features,
                                     class_weight=self.forest_class_weight,
                                     n_jobs=self.forest_n_jobs)
        elif self.algo == 'lda':
            self.method \
            = LinearDiscriminantAnalysis(solver=self.lda_solver,
                                         n_components=self.lda_n_components,
                                         tol=self.lda_tol)
        elif self.is_neural == True:
            layer_name_list = [layer_name for layer_name \
                               in self.__dict__.keys() \
                               if layer_name.startswith('layer') \
                               and layer_name != 'n_layers']
            layer_method_list = []
            for layer_name in layer_name_list:
                layer = getattr(self, layer_name)
                if layer.type == 'dense_feats':
                    layer_method_list \
                    .append(DenseFeatures(feature_columns=self.neural_inputs))
                elif layer.type == 'dense':
                    layer_method_list \
                    .append(Dense(units=layer.units,
                                  activation=layer.activation))
                elif layer.type == 'dropout':
                    layer_method_list \
                    .append(Dropout(rate=layer.rate))
            if self.algo == 'seq':
                self.method = Sequential(layer_method_list)
                self.method.compile(optimizer=self.seq_optimizer,
                                    loss=self.seq_loss,
                                    metrics=['accuracy'])
        return self

    def fit(self,
            data_train,
            feats: list,
            target: str,
            data_valid = None,
            epochs: int = 1,
            seed: int = 1):
        if target is None:
            raise Exception('Unable to apply \'.fit()\' method on None '
                            '\'target\'.')
        if self.is_neural == False:
            self.vector = self.method \
                              .fit(data_train[feats], data_train[target])
        else:
            data_train = DataSet(data=data_train,
                                 feats=feats,
                                 target=target,
                                 shuffle=True,
                                 seed=seed)
            if data_valid is not None:
                data_valid = DataSet(data=data_valid,
                                     feats=feats,
                                     target=target,
                                     shuffle=True,
                                     seed=seed)
            self.history = self.method \
                               .fit(x=data_train,
                                    validation_data=data_valid,
                                    epochs=epochs,
                                    verbose=0)
        return self

    def score(self,
              data_test,
              feats: list,
              target: str,
              seed: int = 1):
        if self.is_neural == False:
            score = self.vector \
                        .score(X=data_test[feats],
                               y=data_test[target])
        else:
            data_test = DataSet(data=data_test,
                                feats=feats,
                                target=target,
                                shuffle=True,
                                seed=seed)
            score = self.method \
                        .evaluate(x=data_test, verbose=0)[1]
        return score

    def cross_validate(self,
                       k: int,
                       data,
                       feats: list,
                       target: str,
                       epochs: int = 1,
                       valid_fold: bool = False,
                       seed: int = 1):
        if k < 3:
            raise ValueError('\'k\' must be greater than or equal to 3.')
        data = data.copy() \
                   .sample(frac=1, random_state=seed)
        data_size = len(data)
        fold_size = data_size // k
        scores = []
        for i in range(0, k):
            data_test_i = data.iloc[fold_size*i: fold_size*(i+1)]
            data_valid_i = data.iloc[fold_size*(i+1): fold_size*(i+2)]
            data_train_i = data.iloc[0: fold_size*i] \
                           .append(data.iloc[fold_size*(i+2): data_size])
            if valid_fold == False:
                data_valid_i = None
                data_train_i = data_train_i.append(data_valid_i)
            score_i = self.fit(data_train=data_train_i,
                               feats=feats,
                               target=target,
                               data_valid=data_valid_i,
                               epochs=epochs) \
                          .score(data_test=data_test_i,
                                 feats=feats,
                                 target=target)
            scores.append(score_i)
        return scores

    def save(self,
             folder: str):
        CheckPath(path=folder, to_be='folder')
        if self.is_neural == True:
            self.method.save(folder+'neural_model')
            self.delete_neural_attr()
        SavePKL(obj=self, path=folder+'model.pkl')
        return self

    def delete_neural_attr(self):
        if self.is_neural == False:
            raise Exception('\'delete_neural_attr\' can only be applied to '
                            'neural models.')
        else:
            neural_attrs = self.__dict__ \
                               .copy()
            for attr in neural_attrs:
                if attr in ['neural_inputs', 'method', 'history'] \
                or attr.startswith('layer'):
                    delattr(self, attr)
        return self

    def apply(self,
              data_test,
              feats: list,
              seed: int = 1):
        if self.is_neural == False:
            data_test[self.long_name] = self.vector.predict(data_test[feats])
        else:
            data_test = DataSet(data=data_test, feats=feats, shuffle=True,
                                seed=seed)
            data_test[self.long_name] = self.method.predict(x=data_test)
        return data_test

# @class_type_check
class Layer():

    def __init__(self,
                 layer_param: dict,
                 name: str):
        self.name = name
        for param in layer_param.keys():
            if type(layer_param[param]) in [str, float, int] \
            or layer_param[param] is None:
                setattr(self, param, layer_param[param])
            else:
                raise TypeError('Parameters must be string, float, '
                                'integer, boolean or NoneType.')
