import boto3
import os
import pandas as pd
import numpy as np
from pipeline import (ccf, ephys, experiment, foraging_analysis,
                      foraging_model, get_schema_name, histology, lab,
                      psth_foraging, report, foraging_analysis_and_export)

bucket = 'aind-behavior-data'
s3_path_root = 'Han/ephys/report/'

local_cache_root = '/root/capsule/results/'



def export_df_foraging_sessions(s3_rel_path='st_cache/', file_name='df_sessions.pkl'):   
    # Currently good for datajoint==0.13.8
    
    foraging_sessions = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol in (100, 110, 120)').proj()
    insertion_numbers = foraging_sessions.aggr(foraging_sessions * ephys.ProbeInsertion, ..., 
                                                    #   keep_all_rows=True, ephys_insertions='IF(COUNT(insertion_number), "yes", "no")')
                                                keep_all_rows=True, ephys_ins='COUNT(insertion_number)')
    if_histology = foraging_sessions.aggr(foraging_sessions * histology.ElectrodeCCFPosition.ElectrodePosition, ...,
                                        keep_all_rows=True, histology='IF(COUNT(ccf_x)>0, "yes", "")')
    if_photostim_from_behav = foraging_sessions.aggr(foraging_sessions * experiment.PhotostimForagingTrial, ...,
                                        keep_all_rows=True, photostim='IF(COUNT(trial)>0, "yes", "")')
    if_photostim_from_ephys = foraging_sessions.aggr(foraging_sessions * (ephys.TrialEvent & 'trial_event_type LIKE "laser%"'), ...,
                                        keep_all_rows=True, photostim_NI='IF(COUNT(trial)>0, "yes", "")')

    df_sessions = pd.DataFrame(((experiment.Session & foraging_sessions)
                                * lab.WaterRestriction.proj(h2o='water_restriction_number',
                                                            weight='wr_start_weight')
                                * lab.Subject.proj('sex')
                                * insertion_numbers
                                * if_histology
                                * if_photostim_from_behav
                                * if_photostim_from_ephys
                                # .proj(..., '-rig', '-username', '-session_time')
                                * lab.Person.proj(user_name='fullname')
                                ).fetch()
                                )

    # df_sessions['session_date'] = pd.to_datetime(df_sessions['session_date'], format="%Y-%m-%d")

    # add task protocol
    df_session_stats = pd.DataFrame((foraging_analysis.SessionStats.proj(
                                                                        finished_trials='session_pure_choices_num', 
                                                                        total_trials = 'session_total_trial_num',
                                                                        foraging_eff='session_foraging_eff_optimal',
                                                                        foraging_eff_randomseed='session_foraging_eff_optimal_random_seed',
                                                                        reward_rate='session_hit_num / session_total_trial_num',
                                                                        miss_rate='session_miss_num / session_total_trial_num',
                                                                        ignore_rate='session_ignore_num / session_total_trial_num',
                                                                        early_lick_ratio='session_early_lick_ratio',
                                                                        double_dipping_ratio='session_double_dipping_ratio',
                                                                        block_num='session_block_num',
                                                                        block_length='session_total_trial_num / session_block_num',
                                                                        mean_reward_sum='session_mean_reward_sum',
                                                                        mean_reward_contrast='session_mean_reward_contrast',
                                                                        autowater_num='session_autowater_num',
                                                                        # session_length_in_hrs='session_length / 3600', # in hrs
                                                                        )
                                * foraging_analysis.SessionTaskProtocol.proj(task='session_task_protocol', not_pretrain='session_real_foraging')
                                * foraging_analysis.SessionEngagementControl.proj(valid_trial_start='start_trial',
                                                                                valid_trial_end='end_trial',
                                                                                valid_ratio='valid_ratio',
                                                                                )
                                & foraging_sessions).fetch())

    df_session_stats.foraging_eff[df_session_stats.foraging_eff_randomseed.notna()] = df_session_stats.foraging_eff_randomseed[df_session_stats.foraging_eff_randomseed.notna()]
    df_session_stats.drop('foraging_eff_randomseed', axis=1, inplace=True)
    df_session_stats['task'].replace({100: 'coupled_block_baiting', 110: 'decoupled_no_baiting', 120: 'random_walk'}, inplace=True)


    # add photostim meta info
    ss = dict(how='left', on=('subject_id', 'session'))
    df_photostim = pd.DataFrame(experiment.PhotostimForagingLocation.fetch())
    df_photostim_trial = pd.DataFrame(experiment.PhotostimForagingTrial.fetch())
    df_photostim = df_photostim.merge(df_photostim_trial.groupby(['subject_id', 'session']).power.median(), **ss
                              ).merge(df_photostim_trial.groupby(['subject_id', 'session'])['bpod_timer_align_to', 'bpod_timer_offset', 'ramping_down'].agg(pd.Series.mode).astype(str), **ss
                              ).merge(df_photostim_trial.groupby(['subject_id', 'session']).size().rename('photostim_trials'), **ss
                              ).merge(df_photostim_trial.groupby(['subject_id', 'session']).trial.agg(lambda x: np.mean(np.diff(x))).rename('photostim_interval_mean'), **ss)
    df_photostim.rename({'power': 'photostim_power_median', 'ramping_down': 'photostim_ramping_down', 'bpod_timer_align_to': 'photostim_aligned_to'}, axis=1, inplace=True)

    # Merge all tables
    df_sessions = df_sessions.merge(df_photostim.query('side == "left"').drop('side', axis=1), how='left', on=('subject_id', 'session')
                                ).merge(df_session_stats, how='left', on=('subject_id', 'session')
                                ).rename(columns={'location': 'photostim_location'})
    
    df_sessions['photostim_trial_ratio'] = df_sessions.photostim_trials / df_sessions.total_trials
        
    # Add headbar info    
    janelia_headbar_allen = ("HH19", "HH20", *[f'KH_FB{n}' for n in np.r_[8, 9, 12:15, 22:25]])
    janelia_headbar_janelia = (lab.WaterRestriction & (foraging_sessions & (experiment.Session & 'session_date < "2022-01-01"'))).fetch('water_restriction_number')
    df_sessions['headbar'] = 'allen_headbar_at_allen'  # by default
    df_sessions.loc[df_sessions.query('h2o in @janelia_headbar_allen').index, 'headbar'] = 'janelia_headbar_at_allen'
    df_sessions.loc[df_sessions.query('h2o in @janelia_headbar_janelia').index, 'headbar'] = 'janelia_headbar_at_janelia'
    
    # Add age in weeks of each session
    df_sessions['age_in_weeks'] = df_sessions.session_date - df_sessions.merge(pd.DataFrame(lab.Subject.proj('date_of_birth').fetch()
                                                                                            ), on='subject_id', how='inner').date_of_birth
    df_sessions.age_in_weeks = df_sessions.age_in_weeks.dt.days / 7
    df_sessions.session_time = df_sessions.session_time / np.timedelta64(1, 'h')
    df_sessions.rename(columns={'session_time': 'time_in_day'}, inplace=True)
                    
    # Remove some bad session
    df_sessions.drop(index=df_sessions.query('h2o == "FOR10" and session == 142').index, inplace=True)
    
    # Bug fix for session length
    session_length = pd.DataFrame(foraging_sessions.aggr(experiment.SessionTrial, session_length_in_hrs='max(start_time)').fetch())
    session_length.length /= 3600
    df_sessions = df_sessions.merge(session_length, on=('subject_id', 'session'), how='left')
    
    # Remove unnecessary columns
    df_sessions.drop(['username'], axis=1, inplace=True)

    # formatting
    # to_int = ['ephys_ins', 'finished', *[col for col in df_sessions if 'num' in col], 'valid_trial_start', 'valid_trial_end', 'total_trials']
    # for col in to_int:
    #     df_sessions[col] = df_sessions[col].astype('Int64')
        
    # to_float = ['foraging_eff', *[col for col in df_sessions if 'rate' in col], 
    #                             *[col for col in df_sessions if 'mean' in col],
    #                             *[col for col in df_sessions if 'ratio' in col], 'length']
    # for col in to_float:
    #     df_sessions[col] = df_sessions[col].astype(float)
    df_sessions = df_sessions.apply(pd.to_numeric, errors='ignore')

        
    # reorder
    #df_sessions = reorder_df(df_sessions, 'h2o', 3)

    for name, order in (('session_date', 0),
                        ('h2o', 1),
                        ('session', 2),
                        ('finished_trials', 4), 
                        ('foraging_eff', 5),
                        ('photostim', 6),
                        ('task', 7),
                        ('time_in_day', 15)
                    ):
        df_sessions = reorder_df(df_sessions, name, order)

    # export and upload
    export_df_and_upload(df_sessions, s3_rel_path, file_name)
    
    return df_sessions


def export_df_model_fitting_param(s3_rel_path='st_cache/'):
    # Add model fitting
    model_ids =  [      8,   # learning rate, e-greedy
                        11,   # tau1, tau2, softmax
                        14,   # Hattori2019, alpha_Rew, alpha_Unr, delta, softmax
                        15,   # 8 + ck
                        17,   # 11 + ck
                        20,   # Hattori 2019 + CK
                        21,   # Hattori 2019 + CK one trial
                 ]
    
    df_model_fitting = pd.DataFrame()
    for model_id in model_ids:
        df_this_model = pd.DataFrame((foraging_model.FittedSessionModel & f'model_id = {model_id}').fetch())
        df_this_model_param = pd.DataFrame((foraging_model.FittedSessionModel.Param & f'model_id = {model_id}').fetch()
                                           ).pivot(index=['subject_id', 'session'], columns='model_param', values='fitted_value')
        df_model_fitting = df_model_fitting.append(df_this_model.merge(df_this_model_param, on=('subject_id', 'session'), how='left'))

    export_df_and_upload(df_model_fitting, s3_rel_path, file_name='df_model_fitting_params.pkl')
    return df_model_fitting


def export_df_regressions(s3_rel_path='st_cache/'):   

    df_linear_regression_rt = pd.DataFrame(foraging_analysis_and_export.SessionLinearRegressionRT.Param.fetch())
    export_df_and_upload(df_linear_regression_rt, s3_rel_path, file_name='df_linear_regression_rt.pkl')

    df_logistic_regression = pd.DataFrame((foraging_analysis_and_export.SessionLogisticRegression & 'trials_back <= 10').fetch())
    export_df_and_upload(df_logistic_regression, s3_rel_path, file_name='df_logistic_regression.pkl')
    
    return

    
# ------- helpers -------

def export_df_and_upload(df, s3_rel_path, file_name):
    # save to local cache
    local_file_name = local_cache_root + file_name
    s3_file_name = s3_path_root + s3_rel_path + file_name

    df.to_pickle(local_file_name)
    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # copy to s3
    res = upload_file(local_file_name, bucket, s3_file_name)
    if res:
        print(f'file exported to {s3_file_name}, size = {size} MB, df_length = {len(df)}')
    else:
        print('Export error!')
    return


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True



def reorder_df(df, column_name, new_loc):
    tmp = df[column_name]
    df = df.drop(columns=[column_name])
    df.insert(loc=new_loc, column=column_name, value=tmp)
    return df