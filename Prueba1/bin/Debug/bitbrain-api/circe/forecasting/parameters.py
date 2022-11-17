# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:00:00 2022

@author: vgini
"""



# %% IMMUTABLE PARAMETERS
# We define the immutable parameters.
# _____________________________________________________________________________

algo_param = {'poly': ['poly_d', 'poly_cycle'],
              'arima': ['arima_p', 'arima_d', 'arima_q'],
              'comp': ['comp_rule', 'comp_seasonal', 'comp_trend',
                       'comp_resid', 'comp_cycle',]}

param_deflt = {'poly_d': 1,
               'poly_cycle': 'weekly',
               'arima_p': 3,
               'arima_d': 1,
               'arima_q': 0,
               'comp_rule': 'add',
               'comp_seasonal': {'algo': 'poly',
                                 'poly_d': 4},
               'comp_trend': {'algo': 'poly',
                              'poly_d': 1},
               'comp_resid': {'algo': 'poly',
                              'poly_d': 0},
               'comp_cycle': 'weekly'}

algo_forbid = {}
