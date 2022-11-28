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


#############################
def gen_random_walk_neuron(n_trial, diff_para=0.1, f_1=2.5, 
                           seed=None, plot=False):
    if seed:
        np.random.seed(seed)

    firing_rate = np.zeros(n_trial)
    firing_rate[0] = np.random.uniform(f_1-0.1, f_1+0.1)
    for i in range(1, n_trial):
        firing_rate[i] = max((0, firing_rate[i-1]+np.random.normal(loc=0, scale=diff_para, size=1)))
    
    if plot:
        plt.plot(firing_rate)

    return firing_rate


def gen_action_value_neuron(qs, k_max_modulation=2.35, f_baseline=2.5, 
                            seed=None, plot=False):
    if seed:
        np.random.seed(seed)
    
    r_neuron_modulation = np.random.uniform(low=-1.0, high=1.0, size=1)
    firing_rate = f_baseline + k_max_modulation * r_neuron_modulation * qs
    spike_cts = np.random.poisson(lam=firing_rate)
    
    if plot:
        plt.plot(spike_cts)

    return spike_cts


############################
if __name__ == "__main__":
    subject_id = 482353
    model_id = 10
    q_latent_variable = (foraging_model.FittedSessionModel.TrialLatentVariable 
                        & {'subject_id': subject_id, 
                        'model_id': model_id})

    df_Q = pd.DataFrame(q_latent_variable.fetch())
    df_Q_left = df_Q[df_Q['water_port']=='left']#.sort_values(by=['trial'])
    df_Q_right = df_Q[df_Q['water_port']=='right']#.sort_values(by=['trial'])

    # get only Qs columns
    df_Q_right = df_Q_right[['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_right'})#.reset_index(drop=True)
    df_Q_left = df_Q_left[['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_left'})#.reset_index(drop=True)
    df_Qs = df_Q_left.merge(df_Q_right)


    # select sessions
    sessions_all = np.unique(df_Qs['session'].values)
    print(f'total sessions: {len(sessions_all)}, {sessions_all}')

    n_session_to_ana = 5
    manual_selection = True
    if manual_selection:
        sessions = [38, 39, 40, 42, 45]
    else:
        sessions = np.random.choice(sessions_all, size=n_session_to_ana)
    print(f'selected sessions for ana: {sessions}')

    # get min session length
    session_len = []
    for session in sessions:
        df_Qs_session = df_Qs[df_Qs['session']==session].sort_values(by=['trial'])
        session_len.append(len(df_Qs_session))

    print(f'min session {session_len.index(min(session_len))} len: {min(session_len)}, max session {session_len.index(max(session_len))} len: {max(session_len)}')
    len_min = min(session_len)


    # generate simulated neurons
    n_neurons = 500
    neuron_types = ['Q_left', 'Q_right', 'sigma_Q', 'delta_Q', 'rw']

    sim_neuron_columns = ['session', 'neuron_id', 'firing_rates']
    sim_neurons = {neuron_type: pd.DataFrame(columns=sim_neuron_columns) for neuron_type in neuron_types}

    for n in range(n_neurons):

        for gen_session in sessions:
            df_Qs_session = df_Qs[df_Qs['session']==gen_session].sort_values(by=['trial'])
            X = df_Qs_session[['Q_left', 'Q_right']]
            print(f'session {gen_session}: {X.shape}')

            # chop the session to chunks of min len
            for i in range(int(np.floor(len(df_Qs_session)/ len_min))):
                t_start = i * len_min

                if i == 0: # remove this to treat different chunks as different sessions
                    for neuron_type in neuron_types:
                        if neuron_type in ['Q_left', 'Q_right']:
                            fr = gen_action_value_neuron(X[neuron_type].values)
                        elif neuron_type == 'sigma_Q':
                            fr = gen_action_value_neuron(X['Q_left'].values + X['Q_right'].values)
                        elif neuron_type == 'delta_Q':
                            fr = gen_action_value_neuron(X['Q_left'].values - X['Q_right'].values)
                        elif neuron_type == 'rw':
                            fr = gen_random_walk_neuron(len(X))
                        else:
                            raise ValueError('wrong neuron types!')
                        sim_neurons[neuron_type].loc[len(sim_neurons[neuron_type].index)] = [gen_session, n, fr]


    # multi_session_permutation fit
    # compute the test statistic: avg t values of regression
    # in all possible permutations

    msp_fit_columns = ['gen_session_perm', 'neuron_id', 'fit_session_perm', 'tvalues_Q_left', 'tvalues_Q_right']
    df_msp_fit_dict = {neuron_type: pd.DataFrame(columns=msp_fit_columns) for neuron_type in neuron_types}

    # fit on all permutations of sessions
    # single neuron, present in all sessions
    for j, p in enumerate(itertools.permutations(range(len(sessions)))):
        print(f'permutation {j}: {p}')

        for n in range(n_neurons):
            for neuron_type in neuron_types:
                neurons = sim_neurons[neuron_type]
                df_msp_fit = df_msp_fit_dict[neuron_type]

                gen_session_perm = []
                fit_session_perm = []
                tvalues_Q_left = []
                tvalues_Q_right = []

                for session_id in range(len(sessions)):
                    gen_session = sessions[session_id]
                    fit_session = sessions[p[session_id]]
                    gen_session_perm.append(gen_session)
                    fit_session_perm.append(fit_session)
                    # print(f' using gen_session {gen_session} and fit_session {fit_session}')

                    # get Qs
                    df_Qs_session = df_Qs[df_Qs['session']==fit_session].sort_values(by=['trial'])
                    X_fit = df_Qs_session[['Q_left', 'Q_right']]
                    X_fit = sm.add_constant(X_fit)
                    X_fit = X_fit[:len_min]
                    # print(f' session {fit_session}: {X_fit.shape}')

                    fr = neurons[(neurons['session']==gen_session) & 
                                (neurons['neuron_id']==n)]['firing_rates'].values[0][:len_min]
                    model = sm.OLS(fr, X_fit)
                    results = model.fit()
                    tvalues_Q_left.append(results.tvalues['Q_left'])
                    tvalues_Q_right.append(results.tvalues['Q_right'])

                tvalues_Q_left = np.array(tvalues_Q_left)
                tvalues_Q_right = np.array(tvalues_Q_right)
                df_msp_fit.loc[len(df_msp_fit.index)] = [gen_session_perm, n, fit_session_perm, 
                                                        tvalues_Q_left, tvalues_Q_right]


    with open('./msp_sim_neurons.pickle', 'wb') as handle:
        pickle.dump(sim_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./df_msp_fit_dict.pickle', 'wb') as handle:
        pickle.dump(df_msp_fit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)