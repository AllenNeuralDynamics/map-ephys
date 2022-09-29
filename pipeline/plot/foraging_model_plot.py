# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""
import pdb
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statannot import add_stat_annotation
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import itertools

from pipeline import lab, foraging_model, util, foraging_analysis, experiment, ephys
from pipeline.plot import behavior_plot, unit_characteristic_plot, unit_psth, histology_plot, PhotostimError, foraging_plot, foraging_model_plot
from pipeline.plot.util import moving_average
from pipeline.report import UnitLevelForagingEphysReportAllInOne
#%%

# plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})
# sns.set(context='talk')


def plot_session_model_comparison(sess_key={'subject_id': 473361, 'session': 47}, model_comparison_idx=0, sort=None):
    """
    Plot model comparison results of a specified session
    :param sess_key:
    :param model_comparison_idx: --> pipeline.foraging_model.ModelComparison
    :param sort: (None) 'aic', 'bic', 'cross_valid_accuracy_test' (descending)
    :return:
    """

    # -- Fetch data --
    results, q_model_comparison = _get_model_comparison_results(sess_key, model_comparison_idx, sort)
    best_aic_id, best_bic_id, best_cross_valid_id = (foraging_model.FittedSessionModelComparison.BestModel & q_model_comparison).fetch1(
        'best_aic', 'best_bic', 'best_cross_validation_test')

    # -- Plotting --
    with sns.plotting_context("notebook", font_scale=1), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(15, 8 + len(results) / 7), dpi=150)
        gs = GridSpec(1, 5, wspace=0.1, bottom=0.11, top=0.85, left=0.33, right=0.98)
        fig.text(0.05, 0.9, f'{(lab.WaterRestriction & sess_key).fetch1("water_restriction_number")}, '
                            f'session {sess_key["session"]}, {results.n_trials[0]} trials\n'
                            f'Model comparison: {(foraging_model.ModelComparison & q_model_comparison).fetch1("desc")}'
                            f' (n = {len(results)})')

        # -- 1. LPT --
        ax = fig.add_subplot(gs[0, 0])
        s = sns.barplot(x='lpt', y='para_notation_with_best_fit', data=results, color='grey')
        s.set_xlim(min(0.5, np.min(np.min(results[['lpt_aic', 'lpt_bic']]))) - 0.005)
        plt.axvline(0.5, color='k', linestyle='--')
        s.set_ylabel('')
        s.set_xlabel('Likelihood per trial')

        # -- 2. aic, bic raw --
        ax = fig.add_subplot(gs[0, 1])
        df = pd.melt(results[['para_notation_with_best_fit', 'aic', 'bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='ic')
        s = sns.barplot(x='ic', y='para_notation_with_best_fit', hue='', data=df)

        # annotation
        x_max = max(plt.xlim())
        ylim = plt.ylim()
        plt.plot(x_max, results.index[results.model_id == best_aic_id] - 0.2, '*', markersize=15)
        plt.plot(x_max, results.index[results.model_id == best_bic_id] + 0.2, '*', markersize=15)
        plt.ylim(ylim)
        s.set_yticklabels('')
        s.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=1)
        s.set_ylabel('')
        s.set_xlabel('AIC or BIC')

        # -- 3. log10_bayesfactor --
        ax = fig.add_subplot(gs[0, 2])
        df = pd.melt(results[['para_notation_with_best_fit', 'log10_bf_aic', 'log10_bf_bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='log10 (bayes factor)')
        s = sns.barplot(x='log10 (bayes factor)', y='para_notation_with_best_fit', hue='', data=df)
        h_d = plt.axvline(-2, color='r', linestyle='--', label='decisive')
        s.legend(handles=[h_d, ], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left')
        plt.ylim(ylim)
        # s.invert_xaxis()
        s.set_xlabel(r'log$_{10}\frac{p(model)}{p(best\,model)}$')
        s.set_ylabel('')
        s.set_yticklabels('')

        # -- 4. model weight --
        ax = fig.add_subplot(gs[0, 3])
        df = pd.melt(results[['para_notation_with_best_fit', 'model_weight_aic', 'model_weight_bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='model weight')
        s = sns.barplot(x='model weight', y='para_notation_with_best_fit', hue='', data=df)
        ax.legend_.remove()
        plt.xlim([-0.05, 1.05])
        plt.axvline(1, color='k', linestyle='--')
        s.set_xlabel('Model weight')
        s.set_ylabel('')
        s.set_yticklabels('')

        # -- 5. Prediction accuracy --
        results.cross_valid_accuracy_test *= 100
        ax = fig.add_subplot(gs[0, 4])
        s = sns.barplot(x='cross_valid_accuracy_test', y='para_notation_with_best_fit', data=results, color='grey')
        plt.axvline(50, color='k', linestyle='--')
        x_max = max(plt.xlim())
        plt.plot(x_max, results.index[results.model_id == best_cross_valid_id], '*', markersize=15, color='grey')
        ax.set_xlim(min(50, np.min(results.cross_valid_accuracy_test)) - 5)
        plt.ylim(ylim)
        ax.set_ylabel('')
        ax.set_xlabel('Prediction accuracy %\n(2-fold cross valid.)')
        s.set_yticklabels('')

    return


def plot_session_fitted_choice(sess_key={'subject_id': 473361, 'session': 47},
                               specified_model_ids=None,
                               model_comparison_idx=0, sort='aic',
                               first_n=1, last_n=0,
                               remove_ignored=True, smooth_factor=5,
                               ax=None):

    """
    Plot actual and fitted choice trace of a specified session
    :param sess_key: could across several sessions
    :param model_comparison_idx: model comparison group
    :param sort: {'aic', 'bic', 'cross_validation_test'}
    :param first_n: top best n competitors to plot
    :param last_n: last n competitors to plot
    :param specified_model_ids: if not None, override first_n and last_n
    :param smooth_factor: for actual data
    :return: axis
    """

    # Fetch actual data
    choice_history, reward_history, iti, p_reward, _ = foraging_model.get_session_history(sess_key, remove_ignored=remove_ignored)
    n_trials = np.shape(choice_history)[1]

    # Fetch fitted data
    if specified_model_ids is None:  # No specified model, plot model comparison
        results, q_model_comparison = _get_model_comparison_results(sess_key, model_comparison_idx, sort)
        results_to_plot = pd.concat([results.iloc[:first_n], results.iloc[len(results)-last_n:]])
    else:  # only plot specified model_id
        results_to_plot = results = _get_specified_model_fitting_results(sess_key, specified_model_ids)
        
    # -- Plot actual choice and reward history --
    with sns.plotting_context("notebook", font_scale=1, rc={'style': 'ticks'}):
        fig, ax = plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor=smooth_factor, ax=ax)

        # -- Plot fitted choice probability etc. --
        model_str =  (f'Model comparison: {(foraging_model.ModelComparison & q_model_comparison).fetch1("desc")}'
                        f'(n = {len(results)})') if specified_model_ids is None else ''
        # plt.gcf().text(0.05, 0.95, f'{(lab.WaterRestriction & sess_key).fetch1("water_restriction_number")}, '
        #                             f'session {sess_key["session"]}, {results.n_trials[0]} trials\n' + model_str)
        
        for idx, result in results_to_plot.iterrows():
            trial, right_choice_prob = (foraging_model.FittedSessionModel.TrialLatentVariable
                                & dict(result) & 'water_port="right"').fetch('trial', 'choice_prob')
            ax.plot(np.arange(0, n_trials) if remove_ignored else trial, right_choice_prob, linewidth=max(1.5 - 0.3 * idx, 0.2),
                    label=f'{idx + 1}: <{result.model_id}>'
                            f'{result.model_notation}\n'
                            f'({result.fitted_param})')

        #TODO Plot session starts
        # if len(trial_numbers) > 1:  # More than one sessions
        #     for session_start in np.cumsum([0, *trial_numbers[:-1]]):
        #         plt.axvline(session_start, color='b', linestyle='--', linewidth=2)
        #         try:
        #             plt.text(session_start + 1, 1, '%g' % model_comparison.session_num[session_start], fontsize=10,
        #                      color='b')
        #         except:
        #             pass

    ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1, 1.3))
    ax.text(0, 1.1, util._get_sess_info(sess_key), fontsize=10, transform=ax.transAxes)
    ax.set_xlabel('Trial number (finished trials only)' if remove_ignored else 'Original trial number')

    # fig.tight_layout()
    # sns.set()

    return ax


def plot_session_lightweight(data, fitted_data=None, smooth_factor=5, base_color='y', ax=None, vertical=False):
    # sns.reset_orig()

    with sns.plotting_context("notebook", font_scale=1):

        choice_history, reward_history, p_reward = data

        # == Fetch data ==
        n_trials = np.shape(choice_history)[1]

        p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

        ignored_trials = np.isnan(choice_history[0])
        rewarded_trials = np.any(reward_history, axis=0)
        unrewarded_trials = np.logical_not(np.logical_or(rewarded_trials, ignored_trials))

        # == Choice trace ==
        fig = None
        if ax is None:  
            fig = plt.figure(figsize=(8, 3), dpi=200)
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.7)

        # Rewarded trials
        xx = np.nonzero(rewarded_trials)[0] + 1
        yy = 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4
        ax.plot(*(xx, yy) if not vertical else [*(yy, xx)], 
                '|' if not vertical else '_', color='black', markersize=10, markeredgewidth=2)

        # Unrewarded trials
        xx = np.nonzero(unrewarded_trials)[0] + 1
        yy = 0.5 + (choice_history[0, unrewarded_trials] - 0.5) * 1.4
        ax.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                '|' if not vertical else '_', color='gray', markersize=6, markeredgewidth=1)

        # Ignored trials
        xx = np.nonzero(ignored_trials)[0] + 1
        yy = [1.1] * sum(ignored_trials)
        ax.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                'x', color='red', markersize=2, markeredgewidth=0.5, label='ignored')

        # Base probability
        xx = np.arange(0, n_trials) + 1
        yy = p_reward_fraction
        ax.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                color=base_color, label='base rew. prob.', lw=1.5)

        # Smoothed choice history
        y = moving_average(choice_history, smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        ax.plot(*(x, y) if not vertical else [*(y, x)],
                linewidth=1.5, color='black', label='choice (smooth = %g)' % smooth_factor)

        # For each session, if any
        if fitted_data is not None:
            ax.plot(np.arange(0, n_trials), fitted_data[1, :], linewidth=1.5, label='model')

        ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1, 1.3))
        
        if not vertical:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Left', 'Right'])
        else:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Left', 'Right'])
    
    # ax.set_xlim(0,300)

    # fig.tight_layout()
    sns.despine(trim=True, ax=ax)

    return fig, ax


def plot_mouse_fitting_results(subject_id, model_comparison_idx=0, sort='aic',
                               model_to_plot_history=14, 
                               para_to_plot_group=[['learn_rate_rew', 'learn_rate_unrew', 'forget_rate'], ['softmax_temperature'], ['biasR']],
                               if_hattori_Fig1I=False):
        
    '''
    Re-implement my legendary code in DJ pipeline for plotting the model fitting results over time of one mouse 
    The original code: https://github.com/hanhou/Dynamic-Foraging/blob/64f628d7beddb783a733540f2b278af59fd2fdf7/utils/run_fit_behavior.py#L191
    
    sort: {'aic', 'bic', 'cross_validation_test'}
    para_to_plot: parameter groups to plot history of
    '''
    
    #%% === Fetching data ===
    h2o = (lab.WaterRestriction & {'subject_id': subject_id}).fetch1('water_restriction_number')
    q_model_comparison = foraging_model.FittedSessionModelComparison() & {'subject_id': subject_id, 'model_comparison_idx': model_comparison_idx}
    model_ids_for_comparison = (foraging_model.ModelComparison.Competitor() & {'model_comparison_idx': model_comparison_idx}
                                ).fetch('model_id', order_by='model_id')
    n_models = len(model_ids_for_comparison)
    sess_stats = pd.DataFrame((q_model_comparison * foraging_analysis.SessionStats()).fetch('session', 'session_pure_choices_num', 
                                                                               'session_foraging_eff_optimal', 
                                                                               'session_foraging_eff_optimal_random_seed',
                                                                               as_dict=True))
    
    # Fill in group_result, same format as my standalone code
    #   SessStats
    group_result = dict()
    group_result['session_number'] = sess_stats['session']
    group_result["n_trials"] = sess_stats['session_pure_choices_num']
    group_result['xlabel'] = [f'{sess} ({trial_num})' for sess, trial_num in zip(sess_stats['session'], sess_stats['session_pure_choices_num'])]
    group_result['foraging_efficiency'] = list(sess_stats['session_foraging_eff_optimal_random_seed'].astype(float))
    if any(np.isnan(group_result['foraging_efficiency'])):
        group_result['foraging_efficiency'] = list(sess_stats['session_foraging_eff_optimal'].astype(float))
    
    #   Pred accuracy
    group_result['session_best'] = (q_model_comparison * foraging_model.FittedSessionModelComparison.BestModel
                                    ).fetch(f'best_{sort}')
    group_result['model_weight_AIC'] = (q_model_comparison * foraging_model.FittedSessionModelComparison.RelativeStat
                                        ).fetch("model_weight_aic", order_by=("model_id", "session")).reshape(n_models, -1)
    
    pred_accu = (q_model_comparison * foraging_model.FittedSessionModel & {'model_id': model_to_plot_history}
                 ).fetch('cross_valid_accuracy_test', 'cross_valid_accuracy_test_bias_only')
    group_result['prediction_accuracy_CV_test'], group_result['prediction_accuracy_CV_test_bias_only'] = pred_accu
    
    #   Fitted parameters
    group_result['fitted_paras'] = {}
    for para_group in para_to_plot_group:
        for para in para_group:
            if para == 'biasR':
                group_result['fitted_paras'][para] = - (q_model_comparison * foraging_model.FittedSessionModel.Param & 
                                                        {'model_id': model_to_plot_history, 'model_param': 'biasL'}
                                                        ).fetch('fitted_value', order_by='session')
            else:
                group_result['fitted_paras'][para] = (q_model_comparison * foraging_model.FittedSessionModel.Param & 
                                                     {'model_id': model_to_plot_history, 'model_param': para}
                                                     ).fetch('fitted_value', order_by='session')
    
        
    # Update notations
    model_notations = (foraging_model.Model & q_model_comparison).fetch('model_notation')
    model_notations = [f'({i}) {m}' for i, m in enumerate(model_notations)]

    #%% ===  Do plotting ===
    # plt.close('all')
    sns.set(context = 'talk')
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})

    # --- 1. Session-wise model weight ---
    fig_model_comp = plt.figure(figsize=(15, 9 / 15 * n_models), dpi = 150)
    gs = GridSpec(1, 20, wspace = 0.1, bottom = 0.15, top = 0.9, left = 0.15, right = 0.95)
    fig_model_comp.text(0.01,0.95,'%s' % (h2o), fontsize = 20)

    ax = fig_model_comp.add_subplot(gs[0: round(20)])
    sns.heatmap(group_result['model_weight_AIC'], annot = True, fmt=".2f", square = False, cbar = False, cbar_ax = [0,1], ax=ax)
    ax.set(xticks= np.r_[0:len(group_result['xlabel'])])
    ax.set_xticklabels(labels=group_result['xlabel'], rotation=-90, ha='left')
    ax.set_yticklabels(model_notations, rotation=0)

    for ss in range(len(group_result["n_trials"])):
        patch = Rectangle((ss, group_result['session_best'][ss]),1,1, color = 'dodgerblue', linewidth = 4, fill= False)
        ax.add_artist(patch)
        
            
    #%% -- 2 Prediction accuracy (cross-validation), and foraging efficiency --
    fig_pred_acc = plt.figure(figsize=(10, 5), dpi = 150)
    gs = GridSpec(1, 1, wspace = 0.2, bottom = 0.15, top = 0.9, left = 0.07, right = 0.95)

    ax = fig_pred_acc.add_subplot(gs[0, 0], sharey = None)    
    plt.axhline(0.5, c = 'k', ls = '--')
    
    x = group_result['session_number']
    marker_sizes = (group_result["n_trials"]/100 * 2)**2

    # # LPT_AIC
    # x = group_result['session_number']
    # y = group_result['LPT_AIC'][overall_best-1,:]
    # plt.plot(x, y, 'b',label = 'likelihood per trial (AIC)', linewidth = 0.7)
    # plt.scatter(x, y, color = 'b', s = marker_sizes, alpha = 0.9)
    # plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], 
    #             facecolors='none', edgecolors = 'b', s = marker_sizes, alpha = 0.7)

    # Prediction accuracy NONCV
    y = group_result['prediction_accuracy_CV_test']
    plt.plot(x, y, 'k',label = 'Pred. acc. 2-fold CV', linewidth = 0.7)
    plt.scatter(x, y, color = 'k', s = marker_sizes, alpha = 0.9)
    
    # Prediction accuracy bias only
    y = group_result['prediction_accuracy_CV_test_bias_only']
    plt.plot(x, y, 'gray', ls = '--', label = 'Pred. acc. bias only 2-fold CV', linewidth = 0.7)
    plt.scatter(x, y, color = 'gray', s = marker_sizes, alpha = 0.9)

    # Foraging efficiency
    y = group_result['foraging_efficiency']
    plt.plot(x, y, 'g', ls = '-', label = 'foraging efficiency', linewidth = 0.7)
    plt.scatter(x, y, color = 'g', s = marker_sizes, alpha = 0.9, marker = '^')
    
    ax.set_ylabel('Likelihood per trial (AIC)')   
    
    plt.title(f'Model: {model_notations[model_to_plot_history]}')
    plt.xlabel('Session number')
    plt.legend()    

    #%% -- 3. Fitted paras over sessions --
    para_plot_color = ('k', 'g', 'r', 'c')
        
    fig_fitted_par = plt.figure(figsize=(15, 5), dpi = 150)
    gs = GridSpec(1, len(para_to_plot_group), wspace = 0.2, bottom = 0.15, top = 0.85, left = 0.05, right = 0.95)

    for pp, ppg in enumerate(para_to_plot_group):
        
        x = group_result['session_number']
        y = np.array([group_result['fitted_paras'][para] for para in ppg])
        
        ax = fig_fitted_par.add_subplot(gs[0, pp : pp + 1], sharey = None)
        ax.set_prop_cycle(color = para_plot_color)
        
        plt.plot(x.T, y.T, linewidth = 0.7)
        
        for ny, yy in enumerate(y):
            ax.scatter(x, yy, s = marker_sizes, alpha = 0.9)

        plt.xlabel('Session number')
        if pp == 0: 
            ax.set_ylabel('Fitted parameters')
        plt.legend([(foraging_model.ModelParam & {'model_param': para}).fetch1('param_notation') for para in ppg], 
                   bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol = 4)    
        plt.axhline(0, c = 'k', ls = '--')
    
    #%% plt.pause(10)      
    return fig_model_comp, fig_pred_acc, fig_fitted_par


def plot_unit_all_in_one(key):
    '''
    All-in-one summary plot for one unit    
    '''
    
    #%%
    # date, imec, unit = '2021-04-18', 0, 541
    # key = (ephys.Unit() & (experiment.Session & 'session_date = "2021-04-18"' & 'subject_id = 473361') & {'insertion_number': imec + 1, 'unit_uid': unit}).fetch1("KEY")

    #%%

    fig = plt.figure(figsize=(60, 50), dpi=100, constrained_layout=0)
    gs0 = fig.add_gridspec(2, 2, height_ratios=(1, 5), width_ratios=(1.3, 1), right=0.95)

    gs_qc = gs0[0, 0].subgridspec(2, 7, hspace=0.4, wspace=0.5)
    gs_drift = gs0[0, 1].subgridspec(2, 3, width_ratios=[6, 1, 1], height_ratios=[1, 11])
    # gs_psth = gs0[1, 0].subgridspec(10, 6, hspace=0.4)
    gs_psth = gs0[1, 0].subgridspec(5, 5, hspace=0.4)

    gs_tuning = gs0[1, 1].subgridspec(5, 2, width_ratios=[4, 1])

    #for ax in (*axs_meta, *axs_psth, *axs_tuning): fig.add_subplot(ax)        


    best_model = (foraging_model.FittedSessionModelComparison.BestModel & key
                  & 'model_comparison_idx=1').fetch1('best_aic')
    align_types = ['trial_start', 'go_cue', 'first_lick_after_go_cue',
                   'iti_start', 'next_trial_start']
    latent_variables = ['relative_action_value_ic', 'total_action_value', 'rpe']
    # === 1. meta info (spike QC etc.) ===
    # Add unit info
    try:
        area_annotation = (((ephys.Unit & key) * histology.ElectrodeCCFPosition.ElectrodePosition) * ccf.CCFAnnotation).fetch1("annotation")
    except:
        area_annotation = 'nan'
    unit_info = (f'{(lab.WaterRestriction & key).fetch1("water_restriction_number")}, '
                f'{(experiment.Session & key).fetch1("session_date")}, '
                f'imec {key["insertion_number"]-1}\n'
                f'Unit #: {key["unit"]}, '
                f'{area_annotation}'
                )
    fig.text(0.1, 0.9, unit_info, fontsize=30)

    # -- mean waveform --
    wave_form = (ephys.Unit & key).fetch1('waveform')
    ts = (1 / (ephys.ProbeInsertion.RecordingSystemSetup & key).fetch1('sampling_rate') * 1000) * range(len(wave_form))

    ax = fig.add_subplot(gs_qc[:1, 0])
    ax.plot(ts, wave_form)
    ax.set(xticks=[0, 1], xlabel='ms', ylabel=R'$\mu V$')
    sns.despine(ax=ax, trim=True)

    # -- spike widths --
    half_width_this_session = (ephys.UnitWaveformWidth & (experiment.Session & key) & UnitLevelForagingEphysReportAllInOne.all_unit_qc
                               ).fetch('waveform_width')
    ax = fig.add_subplot(gs_qc[1, 0])
    ax.hist(half_width_this_session, 30, color='b')
    ax.set(xlabel='Spk widths (this session, ms)')
    ax.axvline((ephys.UnitWaveformWidth & key).fetch1('waveform_width'), color='g', linestyle='-')

    # -- unit QC --
    axs = np.array([fig.add_subplot(gs_qc[row_idx, col_idx])
                    for row_idx, col_idx in itertools.product(range(0, 2), range(1, 4))])

    amp, snr, spk_rate, isi_violation, amplitude_cutoff, presence_ratio = (ephys.Unit * ephys.UnitStat * ephys.ClusterMetric & key).fetch1(
                                        'unit_amp', 'unit_snr', 'avg_firing_rate', 'isi_violation', 'amplitude_cutoff', 'presence_ratio')

    unit_characteristic_plot.plot_clustering_quality_foraging(ephys.ProbeInsertion & key, axs=axs,
                                                              highlight_unit={'amp': amp, 'snr': snr,
                                                                              'isi': np.log10(isi_violation + 1e-5),
                                                                              'rate': np.log10(spk_rate),
                                                                              'amplitude_cutoff': amplitude_cutoff,
                                                                              'presence_ratio': presence_ratio},
                                                              qc_boundary={'amp': 70, 'isi': np.log10(0.5),
                                                                           'amplitude_cutoff': 0.1,
                                                                           'presence_ratio': 0.95}
                                                              )

    # -- unit QC along the probe --
    axs = np.array([fig.add_subplot(gs_qc[:2, col_idx])
                    for col_idx in range(4, 7)])
    unit_characteristic_plot.plot_unit_characteristic(ephys.ProbeInsertion & key, axs=axs, m_scale=500, highlight_unit=key)

    # -- drift map --
    axs = [fig.add_subplot(gs_drift[row_idx, col_idx])
           for row_idx, col_idx in ((1, 0), (0, 0), (1, 1), (1, 2))]
    unit_characteristic_plot.plot_driftmap(ephys.ProbeInsertion & key, axs=axs)


    # === 2. raster & psth ===
    # --- 2.1 choice & outcome ---
    unit_psth.plot_unit_psth_choice_outcome(
        unit_key=key,
        align_types=align_types,
        axs=np.array([fig.add_subplot(gs_psth[row_idx, col_idx])
                      # for row_idx, col_idx in itertools.product(range(2, 4), range(5))]).reshape(2, 5))
                      for row_idx, col_idx in itertools.product(range(0, 2), range(5))]).reshape(2, 5))

    # --- 2.2 deltaQ, sumQ, RPE ---
    # index_range = range(4, 7)
    index_range = range(2, 5)

    for idx, latent_variable in zip(index_range, latent_variables):
        axs = np.array([fig.add_subplot(gs_psth[row_idx, col_idx])
                          for row_idx, col_idx in itertools.product(range(idx, idx+1), range(5))]).reshape(1, 5)
        unit_psth.plot_unit_psth_latent_variable_quantile(
            unit_key=key,
            axs=axs,
            model_id=best_model,
            align_types=align_types,
            latent_variable=latent_variable)
        

    # === 3. period selectivity ===
    independent_variable=['relative_action_value_ic', 'total_action_value', 'rpe']
    axs = {'choice_history': fig.add_subplot(gs_tuning[0, 0]),
           'period_firing': fig.add_subplot(gs_tuning[1, 0]),}
    for n, iv in enumerate(independent_variable):
        axs['time_' + iv] = fig.add_subplot(gs_tuning[2 + n, 0])
        axs['fit_' + iv] = fig.add_subplot(gs_tuning[2 + n, 1])

    unit_psth.plot_unit_period_tuning(unit_key=key,
                                      independent_variable=independent_variable,
                                      model_id=None,  # Best model of this session
                                      axs=axs)
    
    return key


# ---- Helper funcs -----

def _get_model_comparison_results(sess_key, model_comparison_idx=0, sort=None):
    """
    Fetch relevent model comparison results of a specified session
    :param sess_key:
    :param model_comparison_idx:
    :param sort: 'aic', 'bic', 'cross_valid_accuracy_test', etc.
    :return: results in DataFrame, q_model_comparison
    """
    # Get all relevant models
    q_model_comparison = (foraging_model.FittedSessionModelComparison.RelativeStat
                          & sess_key & {'model_comparison_idx': model_comparison_idx})
    q_result = (q_model_comparison
                * foraging_model.Model.proj(..., '-n_params')
                * foraging_model.FittedSessionModel)

    # Add fitted params
    q_result *= q_result.aggr(foraging_model.FittedSessionModel.Param * foraging_model.Model.Param,
                              fitted_param='GROUP_CONCAT(ROUND(fitted_value,2) ORDER BY param_idx SEPARATOR ", ")')
    results = pd.DataFrame(q_result.fetch())
    results['para_notation_with_best_fit'] = [f'<{id}> {name}\n({value})' for id, name, value in
                                              results[['model_id', 'model_notation', 'fitted_param']].values]

    # Sort if necessary
    if sort:
        results.sort_values(by=[sort], ascending=sort != 'cross_valid_accuracy_test', ignore_index=True, inplace=True)

    return results, q_model_comparison


def  _get_specified_model_fitting_results(sess_key, specified_model_ids):
    model_ids_str = ', '.join(str(id) for id in np.array([specified_model_ids]).ravel())
    q_result = (foraging_model.Model.proj(..., '-n_params') * foraging_model.FittedSessionModel
                ) & sess_key & f'model_id in ({model_ids_str})'
    q_result *= q_result.aggr(foraging_model.FittedSessionModel.Param * foraging_model.ModelParam * foraging_model.Model.Param.proj('param_idx'),
                                    fitted_param='GROUP_CONCAT(ROUND(fitted_value,2) ORDER BY param_idx SEPARATOR ", ")')
    return pd.DataFrame(q_result.fetch())
