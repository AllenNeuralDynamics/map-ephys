import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal 
from scipy import stats
import itertools
import seaborn as sns
import statsmodels.api as sm

import os
import random
import datetime

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

from UtilFunctions_KH import _get_independent_variableKH, align_phys_to_behav_trials


if __name__ == "__main__":
    # ps fit, single neuron encoding
    # compute the test statistic: t values of regression with q generation
    # generating pseudo sessions (pseudo Qs)


    # load data
    with open('./ophys_nm_traces.pickle', 'rb') as handle:
        nm_traces_dict = pickle.load(handle)

    with open('./ophys_q_latents.pickle', 'rb') as handle:
        q_latents = pickle.load(handle)
    
    with open('./ophys_pseudo_sessions.pickle', 'rb') as handle:
        df_pseudo_sessions = pickle.load(handle)


    signal_types_to_fit = ['G1', 'R1', 'G2', 'R2']
    resp_types = ['Resp_e', 'Resp_l', 'Resp_t', 'Resp_base']
    q_types = ['Q_left', 'Q_right', 'rpe']

    n_pseudo_sessions = 120
    n_pseudo_sessions_pool = 360

    ps_encode_columns = ['src_session', 'resp_type', 'fit_session', 
                        'tvalues_Q_left', 'tvalues_Q_right', 'tvalues_rpe']
    df_ps_encode_dict = {signal_type: pd.DataFrame(columns=ps_encode_columns) 
                        for signal_type in signal_types_to_fit}


    # fit resps from signals
    for signal_type in signal_types_to_fit:
        print(f'signal type {signal_type}')
        nm_traces_type = nm_traces_dict[signal_type]

        df_Qs = q_latents[signal_type]
        
        df_ps_fit = df_ps_encode_dict[signal_type]


        for session in np.unique(df_Qs['session'].values):
            nm_traces_type_session = nm_traces_type[nm_traces_type['session']==session]
            
            df_Qs_session = df_Qs[df_Qs['session']==session]
            X = np.empty((len(df_Qs_session.iloc[0]['q_time_series']),
                        len(q_types)))
            for j, q_type in enumerate(q_types):
                X[:, j] = df_Qs_session[df_Qs_session['q_type']==q_type].iloc[0]['q_time_series']
            X = sm.add_constant(X)
            print(f' session {session}: {X.shape}')

            df_pseudo_sessions_session = df_pseudo_sessions[df_pseudo_sessions['src_session']==session]


            for resp_type in resp_types:
                fr = nm_traces_type_session[nm_traces_type_session['resp_type']==resp_type].iloc[0]['trace']
                
                # fit true session
                model = sm.OLS(fr, X)
                results = model.fit()
                df_ps_fit.loc[len(df_ps_fit.index)] = [session, resp_type, session, 
                    results.tvalues[1], results.tvalues[2], results.tvalues[3]]

                # fit pseudo sessions
                df_pseudo_sessions_session_resp_type = df_pseudo_sessions_session.sample(n=n_pseudo_sessions)
                for _, row in df_pseudo_sessions_session_resp_type.iterrows():
                    
                    X_pseudo = np.empty((len(df_Qs_session.iloc[0]['q_time_series']),
                                        len(q_types)))
                    for j, q_type in enumerate(q_types):
                        X_pseudo[:, j] = row[f'pseudo_{q_type}'][:-1]
                    X_pseudo = sm.add_constant(X_pseudo)

                    model = sm.OLS(fr, X_pseudo)
                    results = model.fit()
                    df_ps_fit.loc[len(df_ps_fit.index)] = [session, resp_type, -1*(row['gen_id']),
                                                        results.tvalues[1], 
                                                        results.tvalues[2],
                                                        results.tvalues[3]]


    # save the ps_encode df if not existed
    with open('./ophys_ps_encode.pickle', 'wb') as handle:
        pickle.dump(df_ps_encode_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)