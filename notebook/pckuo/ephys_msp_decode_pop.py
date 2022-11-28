import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal 
from scipy import stats
import itertools
import seaborn as sns
import statsmodels.api as sm
import random


import pickle
import json
json_open = open('../../dj_local_conf.json', 'r') 
config = json.load(json_open)

import datajoint as dj
dj.config['database.host'] = config["database.host"]
dj.config['database.user'] = config ["database.user"]
dj.config['database.password'] = config["database.password"]
dj.conn().connect()

from pipeline import lab, get_schema_name, experiment, foraging_model, ephys, foraging_analysis, histology, ccf
from pipeline.plot import unit_psth
from pipeline.plot.foraging_model_plot import plot_session_model_comparison, plot_session_fitted_choice
from pipeline import psth_foraging
from pipeline import util
from pipeline.model import bandit_model






if __name__ == "__main__":

    # load data
    with open('./neurons_data_match_iti_all.pickle', 'rb') as handle:
        neurons = pickle.load(handle)

    with open('./q_latents_match.pickle', 'rb') as handle:
        q_latents = pickle.load(handle)

    with open('./pseudo_sessions_match.pickle', 'rb') as handle:
        pseudo_sessions_dict = pickle.load(handle)


    # msp fit, poppulation decoding
    # compute the test statistic: sum of residuals from regression models
    # in all possible permutations


    regions_to_fit = ['ALM', 'PL', 'ACA', 'ORB', 'LSN', 'striatum', 'MD']
    target_variables = ['Q_left', 'Q_right', 'sigma_Q', 'delta_Q']

    msp_decode_pop_columns = ['gen_session_perm', 'fit_session_perm', 'target_variable', 'residuals', 'mse_total']
    df_msp_decode_pop_dict = {region: pd.DataFrame(columns=msp_decode_pop_columns) for region in regions_to_fit}


    # fit on all permutations of sessions
    for region in regions_to_fit:
        print(f'region {region}')
        neurons_region = neurons[region]
        sessions_with_unit = np.unique(neurons_region['session'].values)

        df_Qs = q_latents[region]
        # calculate minimal session length
        sess_min_len = 100000
        for sess in sessions_with_unit:
            df_Qs_sess = df_Qs[df_Qs['session']==sess].sort_values(by=['trial'])
            sess_len = len(df_Qs_sess)
            sess_min_len = min(sess_min_len, sess_len)
        print(f' min session length: {sess_min_len}')

        df_msp_fit = df_msp_decode_pop_dict[region]


        for j, p in enumerate(itertools.permutations(range(len(sessions_with_unit)))):
            print(f'  permutation {j}: {p}')
        
            for target_variable in target_variables:
                gen_session_perm = []
                fit_session_perm = []
                residuals = []
                mse_total = []

                for session_id in range(len(sessions_with_unit)):
                    gen_session = sessions_with_unit[session_id]
                    fit_session = sessions_with_unit[p[session_id]]
                    gen_session_perm.append(gen_session)
                    fit_session_perm.append(fit_session)
                    # print(f' using gen_session {gen_session} and fit_session {fit_session}')

                    # get population activity
                    neurons_region_session = neurons_region[neurons_region['session']==gen_session]
                    fr = np.empty((sess_min_len, 
                                len(neurons_region_session)))
                    for j in range(len(neurons_region_session)):
                        fr[:, j] = neurons_region_session.iloc[j]['firing_rates'][:sess_min_len]
                    fr = sm.add_constant(fr)
                    #print(f'  sess {gen_session} fr shape {fr.shape}')

                    # get Qs
                    df_Qs_session = df_Qs[df_Qs['session']==fit_session].sort_values(by=['trial'])
                    if target_variable in ['Q_left', 'Q_right']:
                        X = df_Qs_session[[target_variable]][:sess_min_len]
                    elif target_variable == 'sigma_Q':
                        X = (df_Qs_session[['Q_left']].values + df_Qs_session[['Q_right']].values)[:sess_min_len]
                    elif target_variable == 'delta_Q':
                        X = (df_Qs_session[['Q_left']].values - df_Qs_session[['Q_right']].values)[:sess_min_len]
                    else:
                        raise ValueError('incorrect target variable type!')

                    # decoding models: fr --> X
                    model = sm.OLS(X, fr)
                    results = model.fit()
                    #print(f'{neuron_type} {n} {target_variable}')
                    #print(f' {results.f_pvalue}')
                    residuals.append(results.resid)
                    mse_total.append(results.mse_total)
                
                residuals = np.array(residuals)
                mse_total = np.array(mse_total)
                df_msp_fit.loc[len(df_msp_fit.index)] = [gen_session_perm, fit_session_perm, 
                                                        target_variable, residuals, mse_total]


    # save the ps population decoding df if not existed
    with open('./ephys_msp_decode_pop.pickle', 'wb') as handle:
        pickle.dump(df_msp_decode_pop_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)