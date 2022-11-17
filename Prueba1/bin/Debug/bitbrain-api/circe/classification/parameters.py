# -*- coding: utf-8 -*-
'''
Created on Thu Aug  4 13:30:00 2022

@author: vgini
'''



# %% IMMUTABLE PARAMETERS
# We define the immutable parameters.
# _____________________________________________________________________________

algo_param = {'linr': ['linr_n_jobs'],
              'logr': ['logr_c', 'logr_penalty', 'logr_class_weight',
                       'logr_tol', 'logr_max_iter', 'logr_n_jobs'],
              'svc': ['svc_c', 'svc_kernel', 'svc_d', 'svc_gamma',
                      'svc_class_weight', 'svc_tol', 'svc_max_iter'],
              'tree': ['tree_criterion', 'tree_splitter', 'tree_max_depth',
                       'tree_min_split', 'tree_min_leaf', 'tree_max_features',
                       'tree_class_weight'],
              'forest': ['forest_n_estimators', 'forest_criterion',
                         'forest_max_depth', 'forest_min_split',
                         'forest_min_leaf', 'forest_max_features',
                         'forest_class_weight', 'forest_n_jobs'],
              'lda': ['lda_solver', 'lda_n_components', 'lda_tol'],
              'seq': ['seq_optimizer', 'seq_loss']}

param_deflt = {'linr_n_jobs': None,
               'logr_c': 1.0,
               'logr_penalty': 'l2',
               'logr_class_weight': None,
               'logr_tol': 1e-4,
               'logr_max_iter': 100,
               'logr_n_jobs': None,
               'svc_c': 1.0,
               'svc_kernel': 'rbf',
               'svc_d': 3,
               'svc_gamma': 'scale',
               'svc_class_weight': None,
               'svc_tol': 1e-3,
               'svc_max_iter': -1,
               'tree_criterion': 'gini',
               'tree_splitter': 'best',
               'tree_max_depth': None,
               'tree_min_split': 2,
               'tree_min_leaf': 1,
               'tree_max_features': None,
               'tree_class_weight': None,
               'forest_n_estimators': 100,
               'forest_criterion': 'gini',
               'forest_max_depth': None,
               'forest_min_split': 2,
               'forest_min_leaf': 1,
               'forest_max_features': 'auto',
               'forest_class_weight': None,
               'forest_n_jobs': None,
               'lda_solver': 'svd',
               'lda_n_components': None,
               'lda_tol': 1e-4,
               'seq_optimizer': 'adam',
               'seq_loss': 'binary_crossentropy'}

algo_forbid = {}

neural_algos = ['seq']
