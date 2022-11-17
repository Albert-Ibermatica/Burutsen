# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:00:00 2022

@author: vgini
"""



# %% IMMUTABLE PARAMETERS
# We define the immutable parameters.
# _____________________________________________________________________________

algo_param = {'km': ['km_n_clusters', 'km_algo', 'km_n_init', 'km_max_iter',
                     'km_tol'],
              'ap': ['ap_damping', 'ap_distance', 'ap_max_iter'],
              'ac': ['ac_n_clusters', 'ac_distance', 'ac_linkage'],
              'dbscan': ['dbscan_eps', 'dbscan_algo', 'dbscan_distance',
                         'dbscan_n_jobs'],
              'birch': ['birch_n_clusters', 'birch_threshold']}

param_deflt = {'km_n_clusters': 2,
               'km_algo': 'auto',
               'km_n_init': 10,
               'km_max_iter': 300,
               'km_tol': 1e-4,
               'ap_damping': 0.5,
               'ap_distance': 'euclidean',
               'ap_max_iter': 300,
               'ac_n_clusters': 2,
               'ac_distance': 'euclidean',
               'ac_linkage': 'ward',
               'dbscan_eps': 0.5,
               'dbscan_algo': 'auto',
               'dbscan_distance': 'euclidean',
               'dbscan_n_jobs': None,
               'birch_n_clusters': 2,
               'birch_threshold': 0.5}

algo_forbid = {'ac': [{'ac_distance': 'cosine',
                       'ac_linkage': 'ward'},
                      {'ac_distance': 'manhattan',
                       'ac_linkage': 'ward'},
                      {'ac_distance': 'cityblock',
                       'ac_linkage': 'ward'},
                      {'ac_distance': 'l1',
                       'ac_linkage': 'ward'},
                      {'ac_distance': 'l2',
                       'ac_linkage': 'ward'}
                      ]}
