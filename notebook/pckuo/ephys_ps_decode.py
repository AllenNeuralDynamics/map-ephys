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


    # ps fit, single neuron decoding
    # compute the test statistic: t values of regression with q generation
    # generating pseudo sessions (pseudo Qs)

    n_pseudo_sessions = 120
    n_pseudo_sessions_pool = 360

    regions_to_fit = ['ALM', 'PL', 'ACA', 'ORB', 'LSN', 'striatum', 'MD']
    target_variables = ['Q_left', 'Q_right', 'sigma_Q', 'delta_Q']

    ps_decode_columns = ['src_session', 'unit', 'fit_session', 'target_variable', 'ftest_pvalue', 'mse_total']
    df_ps_decode_dict = {region: pd.DataFrame(columns=ps_decode_columns) for region in regions_to_fit}


    # fit neurons from each region
    for region in regions_to_fit:
        print(f'region {region}')
        neurons_region = neurons[region]
        sessions_with_unit = np.unique(neurons_region['session'].values)

        df_Qs = q_latents[region]
        df_pseudo_sessions = pseudo_sessions_dict[region]

        df_ps_fit = df_ps_decode_dict[region]


        for session in sessions_with_unit:
            neurons_region_session = neurons_region[neurons_region['session']==session]

            df_Qs_session = df_Qs[df_Qs['session']==session].sort_values(by=['trial'])
            n_trials = len(df_Qs_session)
            print(f' session {session}: {n_trials}')

            df_pseudo_sessions_session = df_pseudo_sessions[df_pseudo_sessions['src_session']==session]


            for target_variable in target_variables:
                # get true session
                if target_variable in ['Q_left', 'Q_right']:
                    X = df_Qs_session[[target_variable]]
                elif target_variable == 'sigma_Q':
                    X = df_Qs_session[['Q_left']].values + df_Qs_session[['Q_right']].values
                elif target_variable == 'delta_Q':
                    X = df_Qs_session[['Q_left']].values - df_Qs_session[['Q_right']].values
                else:
                    raise ValueError('incorrect target variable type!')
                #print(f' sess {session}, target {target_variable}: {X.shape}')

                for unit in np.unique(neurons_region_session['unit'].values):
                    fr = neurons_region_session[(neurons_region_session['unit']==unit)].iloc[0]['firing_rates']
                    fr = sm.add_constant(fr)

                    # fit true session
                    # decoding models: fr --> X
                    model = sm.OLS(X, fr)
                    results = model.fit()
                    df_ps_fit.loc[len(df_ps_fit.index)] = [session, unit, session, 
                                                        target_variable, 
                                                        results.f_pvalue,
                                                        results.mse_total]

                    # fit pseudo sessions
                    df_pseudo_sessions_session_unit = df_pseudo_sessions_session.sample(n=n_pseudo_sessions)
                    for _, row in df_pseudo_sessions_session_unit.iterrows():
                        X_pseudo = row['q_pseudo']
                        if target_variable == 'Q_left':
                            X_pseudo = X_pseudo[:, 0]
                        elif target_variable == 'Q_right':
                            X_pseudo = X_pseudo[:, 1]
                        elif target_variable == 'sigma_Q':
                            X_pseudo = X_pseudo[:, 0] + X_pseudo[:, 1]
                        elif target_variable == 'delta_Q':
                            X_pseudo = X_pseudo[:, 0] - X_pseudo[:, 1]
                        else:
                            raise ValueError('incorrect target variable type!')

                        # decoding models: fr --> X
                        model = sm.OLS(X_pseudo, fr)
                        results = model.fit()
                        df_ps_fit.loc[len(df_ps_fit.index)] = [session, unit, -1*(row['gen_id']),
                                                            target_variable, 
                                                            results.f_pvalue,
                                                            results.mse_total]


    # save the ps population decoding df if not existed
    with open('./ephys_ps_decode.pickle', 'wb') as handle:
        pickle.dump(df_ps_decode_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)