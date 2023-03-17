import numpy as np
import pandas as pd

import dill
import scipy
import warnings

from to_s3_util import export_df_and_upload


cache_folder = '/root/capsule/data/s3/export/'


z_tuning_mappper = {'dQ_go_cue_before_2': dict(align_to='go_cue', time_win=[-2, 0], latent_name='relative_action_value_lr', latent_variable_offset=-1),
                    'dQ_iti_start_before_1': dict(align_to='iti_start', time_win=[-1, 0], latent_name='relative_action_value_lr', ),
                    'dQ_iti_start_after_2': dict(align_to='iti_start', time_win=[0, 2], latent_name='relative_action_value_lr'),
                    
                    'sumQ_go_cue_before_2': dict(align_to='go_cue', time_win=[-2, 0], latent_name='total_action_value', latent_variable_offset=-1),
                    'sumQ_iti_start_before_1': dict(align_to='iti_start', time_win=[-1, 0], latent_name='total_action_value',),
                    'sumQ_iti_start_after_2': dict(align_to='iti_start', time_win=[0, 2], latent_name='total_action_value'),
                    
                    'rpe_go_cue_before_2': dict(align_to='go_cue', time_win=[-2, 0], latent_name='rpe', latent_variable_offset=-1),
                    'rpe_choice_after_2': dict(align_to='choice', time_win=[0, 2], latent_name='rpe', ),                    
                    'rpe_iti_start_before_1': dict(align_to='iti_start', time_win=[-1, 0], latent_name='rpe',),
                    'rpe_iti_start_after_2': dict(align_to='iti_start', time_win=[0, 2], latent_name='rpe', ),
                   }


def compute_unit_firing_binned_by_latent_variable(df_aligned_spikes,
                                                df_behavior,
                                                align_to='iti_start',
                                                time_win = [0, 2],
                                                latent_name='relative_action_value_lr',
                                                latent_variable_offset=0,   # <=0. e.g., -1 means use the latent variable from trial-1
                                                latent_bins=None,                                                  
                                                if_z_score_latent=True,
                                                ):
    '''
    df_aligned_spikes --> df_unit_latent_bin_firing (raw)
    '''
    
    aligned_spikes = df_aligned_spikes['aligned_firings'][align_to]
    ts = df_aligned_spikes['times'][align_to]

    valid_trials = df_aligned_spikes['trials'][align_to]  # may skip ignored trials
    latent_values = df_behavior.query('trial in @valid_trials')[latent_name]
    choices =  df_behavior.query('trial in @valid_trials')['choice_lr']
    
    # offset latent variable
    assert latent_variable_offset <= 0, 'invalid latent_variable_offset'
    latent_values = latent_values.iloc[:len(latent_values)+latent_variable_offset]
    choices = choices.iloc[:len(choices)+latent_variable_offset]  # For before go cue, this becomes the previous trial
    aligned_spikes = aligned_spikes[:, -latent_variable_offset:, :]

    # z-score latent variable
    if if_z_score_latent:
        latent_values = scipy.stats.zscore(latent_values.astype(float), nan_policy='omit')
        
    # determine bins for latent variable
    if latent_bins is None:
        latent_bins = np.linspace(-2, 2, 20) if if_z_score_latent else np.linspace(0, 1, 20) if 'rpe' not in latent_name else np.linspace(-1, 1, 20)
             
    latent_bin_centers = (latent_bins[:-1] + latent_bins[1:]) / 2
    latent_bins[0] = -np.inf  # Include outliers
    latent_bins[-1] = np.inf

    # compute average firing rate in the given time window (N_neurons * N_trials)
    aligned_aver_firing = np.nanmean(aligned_spikes[:, :, (time_win[0] <= ts) & (ts < time_win[1])], axis=2) / df_aligned_spikes['bin_size']  # spike / s

    # split trials according to the previous and next choice
    choice_mapping = dict(  all_choice = pd.Series(True, index=choices.index),
                            previous_choice_l = choices == 0,  # For before go cue, since it is already offseted, this becomes the previous trial; for after go cue or iti start, this is the trial that just happened
                            previous_choice_r = choices == 1,
                            next_choice_l = (choices == 0).shift(-1).fillna(False),  # choice immediately AFTER 'before go cue', but the choice of the next trial of "after go cue" / "iti start" etc.
                            next_choice_r = (choices == 1).shift(-1).fillna(False))
    
    # for each choice split, compute aver z-score
    dfs = []


    for choice_group, choice_filter in choice_mapping.items():
        # sort firing according to bins of latent variable (N_bins list of N_neurons * N_trials in the bin)
        latent_binned_firing = [aligned_aver_firing[:, (low <= latent_values) & (latent_values < high) & choice_filter] 
                               for low, high in zip(latent_bins[:-1], latent_bins[1:])]
            
        # for each unit, mean and sem across selected trials (N_neurons * N_bins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_mean = pd.DataFrame(np.array(list(map(lambda x: np.nanmean(x, axis=1), latent_binned_firing))).T, 
                                    index=pd.MultiIndex.from_frame(pd.DataFrame(df_aligned_spikes['unit_keys'])), 
                                    columns=pd.MultiIndex.from_tuples([(choice_group, 'mean', center) for center in latent_bin_centers], 
                                                                    names=['choice_group', 'stats', f"{latent_name}{'_z_score' if if_z_score_latent else ''}"]),
                                    )
            df_sem = pd.DataFrame(np.array(list(map(lambda x: scipy.stats.sem(x, axis=1), latent_binned_firing))).T,
                                    index=pd.MultiIndex.from_frame(pd.DataFrame(df_aligned_spikes['unit_keys'])), 
                                    columns=pd.MultiIndex.from_tuples([(choice_group, 'sem', center) for center in latent_bin_centers], 
                                                                    names=['choice_group', 'stats', f"{latent_name}{'_z_score' if if_z_score_latent else ''}"]),
                                    )
        dfs.extend([df_mean, df_sem])
        
    df_unit_latent_bin_firing = pd.concat(dfs, axis=1)

    # for each unit, compute pearson r and p, using all choices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.stats.ConstantInputWarning)
        pearson_r_p = [scipy.stats.pearsonr(x[~np.isnan(x)], latent_bin_centers[~np.isnan(x)])
                    if np.sum(~np.isnan(x)) >= 2
                    else (np.nan, np.nan) 
                    for x in np.array(df_mean)]
    pearson_r, pearson_p = zip(*pearson_r_p)
    df_unit_latent_bin_firing['r'] = pearson_r
    df_unit_latent_bin_firing['p'] = pearson_p
    
    # compute z_score mean and std using average psth aligned to go_cue (spike / s)
    aver_psth = np.mean(df_aligned_spikes['aligned_firings']['go_cue'], axis=1) / df_aligned_spikes['bin_size']
    z_mean = np.mean(aver_psth, axis=1)
    z_std = np.std(aver_psth, axis=1)
    df_unit_latent_bin_firing['z_mean'] = z_mean
    df_unit_latent_bin_firing['z_std'] = z_std
    
    # store meta info
    df_unit_latent_bin_firing._metadata = dict(align_to=align_to, time_win=time_win, latent_name=latent_name, 
                                               latent_variable_offset=latent_variable_offset, if_z_score_latent=if_z_score_latent)
    
    return df_unit_latent_bin_firing


def compute_one_session_z_score(session_key):
    df_aligned_spikes = dill.load(open(cache_folder + f'{session_key["subject_id"]}_{session_key["session"]}_aligned_spike_counts.pkl', 'rb'))
    df_behavior = dill.load(open(cache_folder + f'{session_key["subject_id"]}_{session_key["session"]}_behavior.pkl', 'rb'))
    print(f'{session_key} loaded')
    
    this_session = {'z_score_x': {}, 'raw_x': {}}
    for name, setting in z_tuning_mappper.items():
        # Use z_score latent
        this_session['z_score_x'][name] = compute_unit_firing_binned_by_latent_variable(df_aligned_spikes, df_behavior, **setting)
        
        # Raw latent
        this_session['raw_x'][name] = compute_unit_firing_binned_by_latent_variable(df_aligned_spikes, df_behavior, 
                                                                                    if_z_score_latent=False,
                                                                                    **setting)
    print(f'{session_key} done!')
    
    return this_session



if __name__ == '__main__':
    
    import multiprocessing as mp
    from tqdm import tqdm
    pool = mp.Pool(8)
    
    session_keys =  dill.load(open(cache_folder + 'session_keys.pkl', 'rb'))
    
    all_session = []
    jobs = [pool.apply_async(compute_one_session_z_score, args=(sess,)) for sess in session_keys]
    for job in tqdm(jobs):
        all_session.append(job.get())
    
    # for sess in session_keys:
    #     all_session.append(compute_one_session_z_score(sess))
    
    pool.close()
    pool.join()
      
    for this_setting_name in z_tuning_mappper:
        for x_method in ['z_score_x', 'raw_x']:
            dfs_for_this_setting = []
            for dfs_this_session in all_session:
                dfs_for_this_setting.append(dfs_this_session[x_method][this_setting_name])
                                    
            df_this_setting_all_session = pd.concat(dfs_for_this_setting)
            df_this_setting_all_session._metadata = dfs_this_session[x_method][this_setting_name]._metadata
            fname = f'z_score_all_{this_setting_name}_{x_method}.pkl'
            export_df_and_upload(df_this_setting_all_session, 'export/', fname)