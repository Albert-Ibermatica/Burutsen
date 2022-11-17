# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:45:00 2022

@author: vgini
"""



# %% ENVIRONEMENT
# We load the required libraries and modules.
# _____________________________________________________________________________

import pandas as pd

from statistics import mean
from math import exp, log

from sklearn.cluster import KMeans, AffinityPropagation, \
                            AgglomerativeClustering, DBSCAN, Birch
from sklearn.metrics import davies_bouldin_score, silhouette_score, \
                            calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid

import matplotlib.pyplot as plt
from matplotlib import gridspec

from circe.transformation.methods import DataContext
from circe.clustering.parameters import algo_param, param_deflt, algo_forbid
from circe.clustering.functions import GetGrid
from circe.utilities.typing import class_type_check, strn, intn, floatn, \
                                   listn, dictn, fi, fin, ls, lsn, dlsn



# %% PYTHON CLASSES FOR CLUSTERING
# We define several Python structures for clustering tasks.
# _____________________________________________________________________________

# @class_type_check
class ClusterContext(DataContext):

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
                 target: strn = None):
        if type(data) == DataContext:
            self.__data = data.__data
            self.freq = data.freq
            self.feats = data.feats
            self.__data_norm = data.__data_norm
            self.use_norm = data.use_norm
        else:
            DataContext.__init__(self, data=data, sep=sep, dec=dec,
                                 n_rows=n_rows, columns=columns, header=header,
                                 index=index, time_unit=time_unit,
                                 time_format=time_format, feats=feats)
        self.__n_models = 0
        self.best_model = None

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
            model_i = Model(model_param=model_param_i)
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
               metrics: list = ['db', 'silhouette', 'ch'],
               n_best: intn = 3,
               plot_clusters: bool = False):
        self.metrics = metrics
        self.init_n_models(model_param=model_param)
        model_attr_names = [attr for attr in self.__dict__.keys() \
                            if attr.startswith('__m')]
        self.ranking = pd.DataFrame(columns=['algo', 'params', 'scores',
                                    'scores_avg'])
        self.inference = []
        for model_attr_name in model_attr_names:
            model_i = getattr(self, model_attr_name)
            self.data = model_i.build() \
                               .fit_apply(data=self.data,
                                          feats=self.feats)
            if self.data[model_i.long_name].nunique() > 1:
                self.inference.append(model_i.long_name)
                model_i.scores = model_i.multi_score(data=self.data,
                                                     feats=self.feats,
                                                     metrics=metrics)
                model_i.scores_avg = mean(model_i.scores)
                rank_i = pd.Series({'name': model_i.name,
                                    'long_name': model_i.long_name,
                                    'algo': model_i.algo,
                                    'params': model_i.__dict__,
                                    'scores': model_i.scores,
                                    'scores_avg': model_i.scores_avg},
                                   name=model_i.long_name)
                self.ranking = self.ranking.append(rank_i)
            else:
                delattr(self, model_attr_name)
                self.__n_models -= 1
                print('{} model discarded for producing only one cluster.' \
                      .format(model_i.long_name))
        self.ranking = self.ranking.sort_values(by='scores_avg',
                                                ascending=False)
        if type(n_best) == int:
            self.ranking = self.ranking.iloc[0: n_best, :]
        self.best_model = getattr(self, '__'+self.ranking.index[0])
        self.display()
        if plot_clusters == True:
            self.plot_clusters()
        return self

    def display(self):
        print('Displaying results of {} method(s)...' \
              .format(len(self.ranking)))
        metrics_str = ', '.join(self.metrics)
        scores_str_list = [str(round(score, 2)) for score in self.best_model \
                                                                 .scores]
        scores_str = ', '.join(scores_str_list)
        scores_avg_str = str(round(self.best_model.scores_avg, 2))
        best_param = self.ranking.loc[self.ranking.index[0], 'params']
        param_str_list = ['.{} = {}' \
                          .format(param, best_param[param]) \
                          for param in best_param.keys() \
                          if param not in ['name', 'long_name', 'algo',
                                           'method', 'vector', 'scores',
                                           'scores_avg']]
        param_str = ' \n                  '.join(param_str_list)
        param_str_sizes = [len(param) + 18 for param in param_str_list]
        display1 = '  -Name:          {} \n'.format(self.best_model.name)
        display2 = '  -Algorithm:     {} \n'.format(self.best_model.algo)
        display3 = '  -Distance:      {} \n'.format(self.best_model.distance)
        display4 = '  -Parameters:    {} \n'.format(param_str)
        display5 = '  -Metrics:       {} \n'.format(metrics_str)
        display6 = '  -Scores:        {} \n'.format(scores_str)
        display7 = '  -Average score: {} \n'.format(scores_avg_str)
        if len(self.ranking) > 1:
            alt_str_list = ['#{}: {} (average score: {})' \
                            .format(i+1,
                                    self.ranking.loc[self.ranking.index[i],
                                                     'name'],
                                    round(self.ranking \
                                              .loc[self.ranking.index[i],
                                                   'scores_avg'],
                                          2)) \
                            for i in range(1, len(self.ranking))]
            alt_str = ' \n                  '.join(alt_str_list)
            alt_str_sizes = [len(alt) + 18 for alt in alt_str_list]
            display8 = '  -Alternatives:  {} \n'.format(alt_str)
        else:
            alt_str_sizes = []
            display8 = ''
        max_size = max([len(display1), len(display2), len(display3),
                        len(display5), len(display6), len(display7)] \
                       +param_str_sizes+alt_str_sizes)
        display0 = '-' * (max_size - 1)
        display = '\n' + display0 + '\n' + display1 + display2 + display3 \
                + display4 + display5 + display6 + display7 + display8
        return print(display)

    def plot_clusters(self):
        fig = plt.figure()
        fig.set_size_inches(15, 10)
        (w, h) = GetGrid(n=len(self.ranking))
        grid = gridspec.GridSpec(h, w, figure=fig)
        for i in range(0, len(self.ranking)):
            model_i = getattr(self,
                              '__{}'.format(self.ranking \
                                                .loc[self.ranking.index[i],
                                                     'long_name']))
            self.data.sort_values(by=model_i.long_name, ascending=True,
                                  inplace=True)
            subgrid = grid[i].subgridspec(2, 3, hspace=0.5)
            ax1 = fig.add_subplot(subgrid[0:1, 0:3])
            feats_encoder = {self.feats[i]: 'F{}'.format(i) \
                             for i in range(0, len(self.feats))}
            centroids = model_i.get_centroids(data=self.data,
                                              feats=self.feats) \
                               .rename(mapper=feats_encoder)
            ax1.plot(centroids)
            ax1.set_title('#{}: {}'.format(i+1, model_i.name), loc='left')
            ax1.grid(which='major', linestyle=':', linewidth='0.5',
                     color='black')
            ax1.tick_params(axis='x', which='both', bottom=False,
                            labelrotation=90, labelbottom=True)
            ax2 = fig.add_subplot(subgrid[1:2, 0:2])
            pca = model_i.get_pca(data=self.data, feats=self.feats,
                                  n_components=2)
            for component in pca[model_i.long_name].unique():
                pca_component = pca[pca[model_i.long_name]==component]
                ax2.scatter(x=pca_component['PCA1'],
                            y=pca_component['PCA2'])
            ax2.grid(which='major', linestyle=':', linewidth='0.5',
                      color='black')
            ax2.tick_params(axis='both', which='both', bottom=False,
                            labelbottom=False, left=False, labelleft=False)
            ax3 = fig.add_subplot(subgrid[1:2, 2:3])
            value_counts = self.data[model_i.long_name] \
                               .value_counts(sort=False)
            ax3.pie(x=value_counts, radius=0.7,
                    autopct=lambda x: f'{x:.0f}%' if x > 20 else '',
                    pctdistance=1.6, textprops={'size': 9},
                    explode=[0.1]+[0]*(self.data[model_i.long_name] \
                                           .unique()-1),
                    startangle=90)
        suptitle_str = 'Top {} clustering method(s)'.format(len(self.ranking))
        fig.suptitle(suptitle_str, fontsize=12)
        return self

# @class_type_check
class Model():

    def __init__(self,
                 model_param: dict):
        if type(model_param) is dict:
            if 'algo' in model_param:
                self.algo = model_param['algo']
        elif type(model_param) is str:
            self.algo = model_param
        for param in algo_param[self.algo]:
            setattr(self, param, param_deflt[param])
            if type(model_param) is dict:
                if param in model_param:
                    setattr(self, param, model_param[param])
        self.get_name()

    def get_name(self):
        if self.algo == 'km':
            self.name = 'km:{}:{}'.format(self.km_n_clusters,
                                          self.km_algo)
        elif self.algo == 'ap':
            self.name = 'ap:{}'.format(self.ap_distance)
        elif self.algo == 'ac':
            self.name = 'ac{}:{}'.format(self.ac_n_clusters,
                                         self.ac_distance)
        elif self.algo == 'dbscan':
            self.name = 'dbscan:{}'.format(self.dbscan_distance)
        elif self.algo == 'birch':
            self.name = 'birch:{}'.format(self.birch_n_clusters)
        return self

    def build(self):
        if self.algo == 'km':
            self.method = KMeans(n_clusters=self.km_n_clusters,
                                 algorithm=self.km_algo,
                                 n_init=self.km_n_init,
                                 max_iter=self.km_max_iter,
                                 tol=self.km_tol)
            self.distance = 'euclidean'
        elif self.algo == 'ap':
            self.method = AffinityPropagation(damping=self.ap_damping,
                                              affinity=self.ap_distance,
                                              max_iter=self.ap_max_iter)
            self.distance = self.ap_distance
        elif self.algo == 'ac':
            self.method \
            = AgglomerativeClustering(n_clusters=self.ac_n_clusters,
                                      affinity=self.ac_distance,
                                      linkage=self.ac_linkage)
            self.distance = self.ac_distance
        elif self.algo == 'dbscan':
            self.method = DBSCAN(eps=self.dbscan_eps,
                                 algorithm=self.dbscan_algo,
                                 metric=self.dbscan_distance,
                                 n_jobs=self.dbscan_n_jobs)
            self.distance = self.dbscan_distance
        elif self.algo == 'birch':
            self.method = Birch(n_clusters=self.birch_n_clusters,
                                threshold=self.birch_threshold)
            self.distance = 'euclidean'
        return self

    def fit_apply(self,
                  data,
                  feats: list):
        data[self.long_name] = self.method \
                                   .fit_predict(X=data[feats])
        return data

    def score(self,
              data,
              feats: list,
              metric: str):
        if metric == 'db':
            db = davies_bouldin_score(X=data[feats],
                                      labels=data[self.long_name])
            score = exp(-db)
        elif metric == 'silhouette':
            silhouette = silhouette_score(X=data[feats],
                                          labels=data[self.long_name],
                                          metric=self.distance)
            score = silhouette
        elif metric == 'ch':
            ch = calinski_harabasz_score(X=data[feats],
                                         labels=data[self.long_name])
            score = log(ch)
        return score

    def multi_score(self,
                    data,
                    feats: list,
                    metrics: list):
        if self.long_name not in data.columns:
            raise Exception('\'{}\' inference column is not included in '
                            '\'data\'.'.format(self.long_name))
        scores = []
        for metric in metrics:
            score_i = self.score(data=data,
                                 feats=feats,
                                 metric=metric)
            scores.append(score_i)
        return scores

    def get_pca(self,
                data,
                feats: list,
                n_components: int = 2):
        pca_method = PCA(n_components=n_components)
        pca_array = pca_method.fit_transform(data[feats])
        pca = pd.DataFrame(data=pca_array, columns=['PCA1', 'PCA2'])
        pca[self.long_name] = data.reset_index(drop=True) \
                                  .loc[:, self.long_name]
        self.pca = pca
        return self.pca

    def get_centroids(self,
                      data,
                      feats: list):
        if self.algo in ['km', 'ap']:
            self.centroids = pd.DataFrame(data=self.method.cluster_centers_,
                                          columns=feats) \
                               .T
        else:
            model_centroids = NearestCentroid()
            model_centroids.fit(X=data[feats], y=data[self.long_name])
            self.centroids = pd.DataFrame(model_centroids.centroids_,
                                          columns=feats) \
                               .T
        return self.centroids
