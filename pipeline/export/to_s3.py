import boto3
import os
import pandas as pd
from pipeline import (ccf, ephys, experiment, foraging_analysis,
                      foraging_model, get_schema_name, histology, lab,
                      psth_foraging, report)

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
                                * lab.WaterRestriction.proj(h2o='water_restriction_number')
                                * insertion_numbers
                                * if_histology
                                * if_photostim_from_behav
                                * if_photostim_from_ephys)
                            .proj(..., '-rig', '-username', '-session_time')
                            .fetch()
                                )

    # df_sessions['session_date'] = pd.to_datetime(df_sessions['session_date'], format="%Y-%m-%d")

    # add task protocol
    df_session_stats = pd.DataFrame((foraging_analysis.SessionStats.proj(
                                                                        finished='session_pure_choices_num', 
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
                                                                        length='session_length',
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
    df_photostim = pd.DataFrame(experiment.PhotostimForagingLocation.fetch())
    df_photostim_trial = pd.DataFrame(experiment.PhotostimForagingTrial.fetch())
    df_photostim = df_photostim.merge(df_photostim_trial.groupby(['subject_id', 'session']).power.median(), how='left', on=('subject_id', 'session')
                                    ).merge(df_photostim_trial.groupby(['subject_id', 'session'])['bpod_timer_align_to', 'bpod_timer_offset', 'ramping_down'].agg(pd.Series.mode).astype(str), how='left', on=('subject_id', 'session'))
    df_photostim.rename({'power': 'laser_power_median', 'ramping_down': 'laser_ramping_down', 'bpod_timer_align_to': 'laser_aligned_to'}, axis=1)

    #TODO: laser ratio and median inter_S_interval


    # Merge all tables
    df_sessions = df_sessions.merge(df_photostim.query('side == "left"').drop('side', axis=1), how='left', on=('subject_id', 'session')
                                ).merge(df_session_stats, how='left', on=('subject_id', 'session')
                                ).rename(columns={'location': 'photostim_location'})

    # formatting
    to_int = ['ephys_ins', 'finished', *[col for col in df_sessions if 'num' in col], 'valid_trial_start', 'valid_trial_end']
    for col in to_int:
        df_sessions[col] = df_sessions[col].astype('Int64')
        
    # reorder
    #df_sessions = reorder_df(df_sessions, 'h2o', 3)
    for name, order in (('finished', 4), 
                        ('foraging_eff', 5),
                        ('photostim', 6),
                        ('task', 7),
                    ):
        df_sessions = reorder_df(df_sessions, name, order)



    local_file_name = local_cache_root + file_name
    s3_file_name = s3_path_root + s3_rel_path + file_name

    df_sessions.to_pickle(local_file_name)
    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # copy to s3
    res = upload_file(local_file_name, bucket, s3_file_name)
    if res:
        print(f'file exported to {s3_file_name}, size = {size} MB, df_length = {len(df_sessions)}')
    else:
        print('Export error!')
    
    return df_sessions


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