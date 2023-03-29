import datajoint as dj
import numpy as np
import pathlib
from tqdm import tqdm

from pipeline import lab, experiment, ephys, histology, psth, ccf, psth_foraging
from pipeline.report import get_wr_sessdatetime

report_cfg = dj.config['stores']['report_store']
store_stage = pathlib.Path(report_cfg['stage'])
        
def export_foraging_behavior():
    ''' 
    Export all foraging sessions (for Faeze)
    
    The dictionary in each .npy file uses session number as the key
    {1: {'choice_history': 1-d array, 0 = left, 1 = right, nan = ignored
         'reward_history': 2-d array, first dimension: left [0] / right [1], second dimension: trial number. 0 = no reward, 1 = rewarded
         'p_reward': underlying reward refill probabilities. Same structure as reward_history
         'trial_num': total trial number (including ignored trials)
         'foraging_efficiency': overall performance of this session
         },
     2: data for session 2,
     ...
    }
    '''

    from pipeline.foraging_model import get_session_history
    from pipeline import foraging_analysis, util
    from pipeline.plot import foraging_model_plot  
    import matplotlib.pyplot as plt

    export_path = pathlib.Path('/root/capsule/data/behavior_export')

    # %%
    foraging_sessions = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol in (100, 110, 120)') * lab.WaterRestriction  # Two-lickport foraging
    
    all_foraging_subject = (dj.U('water_restriction_number') & foraging_sessions).fetch('KEY')

    for subject_key in all_foraging_subject:
        sessions_this_subject = (foraging_sessions & subject_key).fetch('KEY')
        h2o = (lab.WaterRestriction & subject_key).fetch1('water_restriction_number')
        
        pathlib.Path(f'{export_path}/{h2o}/').mkdir(parents=True, exist_ok=True)
        # this_subject = dict()
            
        for session_key in sessions_this_subject:
            
            datestr = (experiment.Session & session_key).fetch1('session_date').strftime(r"%Y%m%d")
            file_prefix = f'{h2o}_{datestr}_{session_key["session"]}'
            file_full_path = export_path / h2o /f'{file_prefix}.npy'

            # Skip if npy already exists
            if file_full_path.exists():
                print(f'Skip {file_prefix} because .npy already exists...')
                continue        

            choice_history, reward_history, _ , p_reward, _ = get_session_history(session_key, remove_ignored=False)
            # try:
            #     trial_num, foraging_efficiency = (foraging_analysis.SessionStats & session_key).fetch1('session_total_trial_num', 'session_foraging_eff_optimal')
            # except:
            #     print(f'Error in fetching foraging_efficiency for {h2o}, {session_key}!!')
            #     continue
            
            # fig, ax = foraging_model_plot.plot_session_lightweight([choice_history, reward_history, p_reward])  # Include ignored trials
            # # foraging_model_plot.plot_session_fitted_choice(session_key)    
            # ax.text(0, 1.1, util._get_sess_info(session_key), fontsize=10, transform=ax.transAxes)
            # fig.savefig(f'./report/behavior/{h2o}/{h2o}_Session_{session_key["session"]:02}')
            
            this_subject = {'choice_history': choice_history[0],
                                                    'reward_history': np.sum(reward_history, axis=0),
                                                    'p_reward': p_reward,
                                                    # 'trial_num': trial_num,
                                                    # 'foraging_efficiency': foraging_efficiency
                                                    }
            
            np.save(file_full_path, this_subject, allow_pickle=True)
            
            plt.close()
            print(f'Done {h2o}, {session_key}')
           
            
        

def write_to_activity_viewer_foraging(insert_keys, output_dir='./'):
    """
    :param insert_keys: list of dict, for multiple ProbeInsertion keys
    :param output_dir: directory to write the npz files
    """

    if not isinstance(insert_keys, list):
        insert_keys = [insert_keys]

    for key in insert_keys:
        water_res_num, sess_datetime = get_wr_sessdatetime(key)

        uid = f'{water_res_num}_{sess_datetime}_{key["insertion_number"]}'

        if not (ephys.Unit * lab.ElectrodeConfig.Electrode * histology.ElectrodeCCFPosition.ElectrodePosition & key):
            print(f'The units in {uid} do not have CCF data yet')
            continue
        
        if not (ephys.ClusterMetric & key):
            print(f'The units in {uid} do not have ClusterMetric!!')
            continue

        q_unit = (ephys.UnitStat * ephys.Unit * lab.ElectrodeConfig.Electrode * ephys.ClusterMetric
                  * histology.ElectrodeCCFPosition.ElectrodePosition  & key
                  & 'presence_ratio > 0.95'
                  & 'amplitude_cutoff < 0.1'
                  & 'isi_violation < 0.5' 
                  & 'unit_amp > 70'
                  ) # & 'unit_quality != "all"' & 'trial_condition_name in ("good_noearlylick_hit")')
        
        if not q_unit:
            print(f'Other problem with {uid} of {key}...')
            continue

        # --- Model-independent stats ---
        unit_id, ccf_x, ccf_y, ccf_z, unit_amp, unit_snr, avg_firing_rate, isi_violation, waveform = q_unit.fetch(
            'unit', 'ccf_x', 'ccf_y', 'ccf_z', 'unit_amp', 'unit_snr', 'avg_firing_rate', 'isi_violation', 'waveform', order_by='unit')
        unit_stats = ["unit_amp", "unit_snr", "avg_firing_rate", "isi_violation"]
        
        # --- Model-dependent stats ---
        q_unit *= psth_foraging.UnitPeriodLinearFit * psth_foraging.UnitPeriodLinearFit.Param

        if not q_unit:
            print(f'Foraging-stats-related problem with {uid} of {key}...')
            continue
                
        # paras to export
        behavior_models = ['best_aic']
        multi_linear_models = {0: 'Q_c + Q_i + rpe', 1: 'Q_rel + Q_tot + rpe'}
        var_names = {0: {0: 'contra_action_value', 1: 'ipsi_action_value', 2: 'rpe'},
                     1: {0: 'relative_action_value_ic', 1: 'total_action_value', 2: 'rpe'}}
        stat_names = ['beta', 'p', 't']
        period_names =  ['before_2', 'delay', 'go_to_end', 'go_1.2', 'iti_all', 'iti_first_2', 'iti_last_2']
        
        time_series_names = ['before_2', 'delay', 'go_to_end', 'iti_first_2', 'iti_all', 'iti_last_2']
        
        foraging_stats = {}
        foraging_time_series = {}
        
        for bm in behavior_models:
            for ind_mlm, multi_linear_model in multi_linear_models.items():
                for ind_vn, var_name in var_names[ind_mlm].items():
                    for stat_name in stat_names:
                        this_psth = []

                        for period_name in period_names:
                            this_name = f'{ind_mlm}_{var_name}_{stat_name}_{period_name}'
                            this_value = (
                                q_unit & {'behavior_model': bm, 'multi_linear_model': multi_linear_model, 
                                          'var_name': var_name, 'period': period_name}
                                ).fetch(stat_name, order_by='unit')
                            
                            # - to psth - ([1 + num_cell, num_time_point])
                            if period_name in time_series_names:
                                this_psth.append(this_value)          
                                
                            # - to unit_stats for filtering -
                            foraging_stats[this_name] = this_value
                            unit_stats.append(this_name)
                                
                            # - add absolute value for t-value
                            if stat_name == 't':
                                tmp_name = f'{ind_mlm}_{var_name}_|t|_{period_name}'
                                foraging_stats[tmp_name] = np.abs(this_value)
                                unit_stats.append(tmp_name)
                                                        
                        # pack and add time to psth
                        foraging_time_series[f'{ind_mlm}_{var_name}_{stat_name}'] = np.vstack([np.r_[0:len(time_series_names)], np.vstack(this_psth).T])
                        if stat_name == 't':
                            foraging_time_series[f'{ind_mlm}_{var_name}_|t|'] = np.vstack([np.r_[0:len(time_series_names)], np.abs(np.vstack(this_psth)).T])
                            
                        
        # --- Wrap up ---
        timeseries = list(foraging_time_series.keys())
        waveform = np.stack(waveform)
        # spike_rates = np.array([d[0] for d in unit_psth])
        # edges = unit_psth[0][1] if unit_psth[0][1].shape == unit_psth[0][0].shape else unit_psth[0][1][1:]
        # unit_psth = np.vstack((edges, spike_rates))
        unit_psth = np.zeros([len(unit_id)+1, 100])# Fake psth
        ccf_coord = np.transpose(np.vstack((ccf_z, ccf_y, 11400 - ccf_x)))

        filepath = pathlib.Path(output_dir) / uid

        print(f'saving to {filepath}...', end='')
        np.savez(filepath, probe_insertion=uid, unit_id=unit_id, ccf_coord=ccf_coord, waveform=waveform,
                 timeseries=timeseries, unit_stats=unit_stats, unit_amp=unit_amp, unit_snr=unit_snr,
                 avg_firing_rate=avg_firing_rate, isi_violation=isi_violation, 
                 **foraging_stats, **foraging_time_series,
                 unit_psth=unit_psth)
        print('done!')