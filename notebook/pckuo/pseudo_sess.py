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


#########################
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


def get_block_lengths(p_reward_trial, p_reward_block):
    block_switch_trial = [0]
    for block_ind in range(p_reward_block.shape[1]):
        trials = np.where((p_reward_trial[0] == p_reward_block[0, block_ind]) & 
                          (p_reward_trial[1] == p_reward_block[1, block_ind]) &
                          [False if t<block_switch_trial[-1] else True for t in range(p_reward_trial.shape[1])])[0]
        # print(f'{block_ind}: {trials}')
        # print(f' {np.where(np.diff(trials) > 1)}')
        if len(np.where(np.diff(trials) > 1)[0])>0:
            trials_first_switch_ind = np.where(np.diff(trials) > 1)[0][0] + 1
            block = trials[: trials_first_switch_ind]
        else:
            block = trials
        # print(f' {block}')
        block_switch_trial.append(block[-1]+1)
    
    block_switch_trial = np.array(block_switch_trial)
    block_lengths = np.diff(block_switch_trial)

    return block_lengths, block_switch_trial


def gen_p_reward_sequence(p_reward_trial, p_reward_block, block_lengths, 
                          block_generator_spread=5, seed=None, plot=False):
    
    p_reward_gen = np.empty((p_reward_trial.shape[0], p_reward_trial.shape[1]+1))
    #print(f'p_reward_block shape: {p_reward_block.shape}')
    #print(f'p_reward_gen shape: {p_reward_gen.shape}')

    # generate new block lengths
    if seed:
        np.random.seed(seed)
    
    not_all_positive = True
    while not_all_positive:
        block_length_generator = np.rint(np.random.uniform(
                                            -1*block_generator_spread+0.1, 
                                            block_generator_spread+0.1, 
                                            size=len(block_lengths)-1)).astype(np.int)
        block_length_generator = np.append(block_length_generator, -1*np.sum(block_length_generator))
        block_lengths_gen = (block_lengths + block_length_generator).astype(np.int)
        not_all_positive = np.any(block_lengths_gen<=0)
        
    #print(f'original block lengths: {block_lengths}')
    #print(f'generated new block length: {block_lengths_gen}')
    #print(f'check total trial number: {np.sum(block_lengths_gen)}')
    
    # generate new p_reward arr
    for i in range(p_reward_trial.shape[0]):
        for block_gen_ind in range(len(block_lengths_gen)):
            if block_gen_ind == 0:
                p_reward_gen[i, 0:block_lengths_gen[block_gen_ind]] = p_reward_block[i, block_gen_ind]
            else:
                t_start = np.sum(block_lengths_gen[:block_gen_ind])
                p_reward_gen[i, t_start:t_start+block_lengths_gen[block_gen_ind]] = p_reward_block[i, block_gen_ind]
        p_reward_gen[i, -1] = p_reward_gen[i, -2]

    if plot:
        sides = ['left', 'right']
        colors = ['r', 'b']
        plt.figure(figsize=(8,4), dpi=150)
        for i in range(p_reward_trial.shape[0]):
            plt.plot(p_reward_trial[i], color=colors[i], label=f'original {sides[i]}')
            plt.plot(p_reward_gen[i], color=colors[i], linestyle='--', label=f'generated {sides[i]}')
            plt.title(f'Generated reward rate blocks')
            plt.xlabel('trial')
            plt.ylabel('reward_rate')
        plt.legend(bbox_to_anchor=(1, 0.6))
        plt.tight_layout()
            
    return p_reward_gen, block_lengths_gen


def gen_qs_by_fitted_model(key, p_reward_trial, p_reward_block, block_lengths, 
                           gen_type='random_block_struct', block_generator_spread=5, forager_type='LNP_softmax'):
    
    # gen_type: ['random_block_struct', 'same_block_struct']

    # get fitted model
    model_para = pd.DataFrame((foraging_model.FittedSessionModel.Param & key).fetch())
    model_para_dict = {}
    for i in range(len(model_para)):
        model_para_dict[model_para.iloc[i]['model_param']] = model_para.iloc[i]['fitted_value']
        
    forager = bandit_model.BanditModel(forager=forager_type, n_trials=int((p_reward_trial).shape[1]), 
                                    **model_para_dict)
    #for k, v in vars(forager).items():
    #    if v is not None:
    #        print(f'{k}: {v}')
    if gen_type == 'random_block_struct':
        block_length_avg = np.average(block_lengths)
        block_length_std = np.std(block_lengths)

        forager.generate_p_reward(block_length_avg, block_length_std)
        print('simulating forager')
        forager.simulate()
    
    
    elif gen_type == 'same_block_struct':
        # generate reward sequence
        p_reward_gen, block_lengths_gen = gen_p_reward_sequence(p_reward_trial, p_reward_block, block_lengths, 
                                                                block_generator_spread=block_generator_spread, plot=False)
        forager.n_blocks = len(block_lengths)
        forager.p_reward = p_reward_gen
        forager.block_size = block_lengths_gen
        forager.p_reward_fraction = p_reward_gen[1]/ np.sum(p_reward_gen, axis=0)
        forager.p_reward_ratio = p_reward_gen[1]/ p_reward_gen[0]
        #for k, v in vars(forager).items():
        #    if v is not None:
        #        print(f'{k}: {v}')

        # generative mode
        # Initialization
        forager.time = 0

        # All latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        forager.q_estimation = np.full([forager.K, forager.n_trials+1], np.nan)
        forager.q_estimation[:, 0] = 0

        forager.choice_prob = np.full([forager.K, forager.n_trials+1], np.nan)
        forager.choice_prob[:, 0] = 1/forager.K   # To be strict (actually no use)

        forager.choice_history = np.zeros(
                        [1, forager.n_trials+1], dtype=int)  # Choice history
        # Reward history, separated for each port (Corrado Newsome 2005)
        forager.reward_history = np.zeros([forager.K, forager.n_trials+1])

        forager.reward_available = np.zeros([forager.K, forager.n_trials+1])
        forager.reward_available[:, 0] = (np.random.uniform(0, 1, forager.K) < forager.p_reward[:, forager.time]).astype(int)

        if forager.forager in ['LNP_softmax', 'LNP_softmax_CK']:
            # Compute the history filter. Compatible with any number of taus.
            # Use the full length of the session just in case of an extremely large tau.
            reversed_t = np.flipud(np.arange(forager.n_trials+1))
            forager.history_filter = np.zeros_like(reversed_t).astype('float64')
            for tau, w_tau in zip(forager.taus, forager.w_taus):
                # Note the normalization term (= tau when n -> inf.)
                forager.history_filter += w_tau * \
                                        np.exp(-reversed_t / tau) / \
                                        np.sum(np.exp(-reversed_t / tau))

        for t in range(forager.n_trials):
            action = forager.act()
            forager.step(action)
        #for k, v in vars(forager).items():
        #    if v is not None:
        #        print(f'{k}: {v}')
    
    else:
        raise ValueError('wrong gen_type for q generation')

    return forager.q_estimation


###############################################
if __name__ == "__main__":

    # generate session action-reward sequence
    key = {'subject_id': 482353, 'session': 42, 'model_id': 10}
    sessions = [key['session']]

    # get fitted q variables
    q_latent_variable = (foraging_model.FittedSessionModel.TrialLatentVariable & key)

    df_Q = pd.DataFrame(q_latent_variable.fetch())
    df_Q_left = df_Q[df_Q['water_port']=='left']#.sort_values(by=['trial'])
    df_Q_right = df_Q[df_Q['water_port']=='right']#.sort_values(by=['trial'])

    # get only Qs columns
    df_Q_right = df_Q_right[['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_right'})#.reset_index(drop=True)
    df_Q_left = df_Q_left[['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_left'})#.reset_index(drop=True)
    df_Qs = df_Q_left.merge(df_Q_right)


    # get session history
    p_reward_block_df = pd.DataFrame((experiment.SessionBlock.WaterPortRewardProbability() & key).fetch())
    p_reward_block_left = p_reward_block_df[p_reward_block_df['water_port']=='left'].sort_values(by='block')['reward_probability'].values
    p_reward_block_right = p_reward_block_df[p_reward_block_df['water_port']=='right'].sort_values(by='block')['reward_probability'].values
    p_reward_block = np.vstack((p_reward_block_left, p_reward_block_right)).astype(np.float)

    blocks = np.unique(p_reward_block_df['block'].values)
    if np.max(blocks) != len(blocks):
        raise ValueError('missing blocks!')
    else:
        n_blocks = len(blocks)

    choice_history, reward_history, _, p_reward_trial, q_choice_outcome = foraging_model.get_session_history(key, remove_ignored=True)
    block_lengths, _ = get_block_lengths(p_reward_trial, p_reward_block)


    # generate simulated neurons
    n_neurons = 120
    neuron_types = ['Q_left', 'Q_right', 'sigma_Q', 'delta_Q', 'rw']

    sim_neuron_columns = ['session', 'neuron_id', 'firing_rates']
    sim_neurons = {neuron_type: pd.DataFrame(columns=sim_neuron_columns) for neuron_type in neuron_types}

    for n in range(n_neurons):

        for gen_session in sessions:
            df_Qs_session = df_Qs[df_Qs['session']==gen_session].sort_values(by=['trial'])
            X = df_Qs_session[['Q_left', 'Q_right']]
            #print(f'session {gen_session}: {X.shape}')

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


    # compute the test statistic: t values of regression with q generation
    # generating pseudo sessions (pseudo Qs)

    n_pseudo_sessions = 120

    # pseudo_session fit
    ps_fit_columns = ['gen_session', 'neuron_id', 'fit_session', 'tvalues_Q_left', 'tvalues_Q_right']
    # if fit_session<0: fit on pseudo sessions
    df_ps_fit_dict = {neuron_type: pd.DataFrame(columns=ps_fit_columns) for neuron_type in neuron_types}

    # fit simulated neurons on pseudo_sessions
    for session in sessions:
        # get session history
        key = {'subject_id': 482353, 'session': session, 'model_id': 10}
        choice_history, reward_history, _, p_reward_trial, q_choice_outcome = foraging_model.get_session_history(key, remove_ignored=True)
        p_reward_block_df = pd.DataFrame((experiment.SessionBlock.WaterPortRewardProbability() & key).fetch())
        p_reward_block_left = p_reward_block_df[p_reward_block_df['water_port']=='left'].sort_values(by='block')['reward_probability'].values
        p_reward_block_right = p_reward_block_df[p_reward_block_df['water_port']=='right'].sort_values(by='block')['reward_probability'].values
        p_reward_block = np.vstack((p_reward_block_left, p_reward_block_right)).astype(np.float)
        block_lengths, _ = get_block_lengths(p_reward_trial, p_reward_block)
        
        for neuron_type in neuron_types:
            neurons = sim_neurons[neuron_type]
            df_ps_fit = df_ps_fit_dict[neuron_type]
            
            for n in np.unique(neurons['neuron_id'].values).astype(np.int):
                
                # fit true sessions
                df_Qs_session = df_Qs[df_Qs['session']==session].sort_values(by=['trial'])
                X = df_Qs_session[['Q_left', 'Q_right']]
                X = sm.add_constant(X)

                fr = neurons[(neurons['session']==session) & (neurons['neuron_id']==n)]['firing_rates'].values[0]
                model = sm.OLS(fr, X)
                results = model.fit()
                df_ps_fit.loc[len(df_ps_fit.index)] = [session, n, session, results.tvalues['Q_left'], results.tvalues['Q_right']]

                # fit pseudo sessions
                for j in range(n_pseudo_sessions):
                    print(f'stating generating {j}')
                    X = gen_qs_by_fitted_model(key, p_reward_trial, p_reward_block, block_lengths).T[:-1, :]
                    #print(f'pseudo session {j} generated')
                    X = sm.add_constant(X)
                    model = sm.OLS(fr, X)
                    results = model.fit()
                    #print(f'pseudo session {j} fitted')
                    df_ps_fit.loc[len(df_ps_fit.index)] = [session, n, -1*(j+1), results.tvalues[1], results.tvalues[2]]
            print(f'end of neuron type {neuron_type}')
            
        print(f'end of session {session}: {X.shape}')


    with open('./ps_sim_neurons_sess_42_n_120_ps_120.pickle', 'wb') as handle:
        pickle.dump(sim_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./df_ps_fit_dict_sess_42_n_120_ps_120.pickle', 'wb') as handle:
        pickle.dump(df_ps_fit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)