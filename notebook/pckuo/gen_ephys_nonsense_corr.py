import numpy as np
import pandas as pd

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


########################
# select units
def select_unit_qc_region_mouse_session(all_unit_qc, region_annotation="Prelimbic%", mouse="HH09", session=47):
    unit_qc_region = (all_unit_qc * histology.ElectrodeCCFPosition.ElectrodePosition * ccf.CCFAnnotation) & 'annotation LIKE "{}"'.format(region_annotation)
    return unit_qc_region & (lab.WaterRestriction & 'water_restriction_number = "{}"'.format(mouse)) & 'session = {}'.format(session)
    

# fetch keys
def gen_keys2units(unit_qc_region_mouse_session):
    # get all keys
    keys2units = unit_qc_region_mouse_session.fetch('KEY')
    print('num of units: {}'.format(len(keys2units)))

    return keys2units


# fetch data from brain region
region_ann_lut = {
    # premotor
    'ALM': "Secondary motor area%",
    # isocortex, PFC
    'PL': "Prelimbic%",
    'ACA': "Anterior cingulate area%",
    'ILA': "Infralimbic%",
    'ORB': '%orbital%',
    'FRP': '%frontal%',
    'RSP': "Retrosplenial area%",
    # thalamus
    'VM': 'Ventral medial%',
    'MD': 'Mediodorsal%',
    # striatum
    'LSN': "Lateral septal nucleus%",
    'CP': "Caudoputamen%",
    'NA': "Nucleus accumbens%",
    'striatum': "striatum%",
    # Pallidum
    'PALv': "Substantia innominata%",
    # Olfactory
    'OLF': "%olfactory%",
}


###############
if __name__ == "__main__":
    # fetch unit data
    # after unit qc
    foraging_session = experiment.Session & 'username = "hh"'
    all_unit_qc = (ephys.Unit * ephys.ClusterMetric * ephys.UnitStat) & foraging_session & 'presence_ratio > 0.95' & 'amplitude_cutoff < 0.1' & 'isi_violation < 0.5' & 'unit_amp > 70'
    dj.U('annotation').aggr(((ephys.Unit & all_unit_qc.proj()) * histology.ElectrodeCCFPosition.ElectrodePosition) * ccf.CCFAnnotation, count='count(*)').fetch(format='frame', order_by='count desc')[:]


    # fetch and organize data

    period = 'iti_all'
    # periods: experiment.PeriodForaging

    model_id = 10

    # use only latter sessions
    latter_session = 30

    regions = ['ALM', 'PL', 'ACA', 'ORB', 'LSN', 'striatum', 'MD']
    mouses = {
        'ALM': 'HH13', #sess=4
        'PL': 'HH08', #sess=4
        'ACA': 'HH13', #sess=5
        'ORB': 'HH09', #sess=4
        'LSN': 'HH13', #sess=6
        'striatum': 'HH08', #sess=5
        'MD': 'HH09' #sess=4
    }

    neuron_columns = ['session', 'unit', 'firing_rates']
    neurons = {region: pd.DataFrame(columns=neuron_columns) for region in regions}

    q_latent_columns = ['session', 'Q_left', 'Q_right']
    q_latents = {region: None for region in regions}


    for region in regions:
        mouse = mouses[region]
        subject_id = (lab.WaterRestriction & f'water_restriction_number="{mouse}"').fetch('subject_id')[0]
        print(f'region: {region}')
        
        # get fitted latent variables
        q_latent_variable = (foraging_model.FittedSessionModel.TrialLatentVariable & 
                            {'model_id': model_id, 'subject_id': subject_id})
        df_Q = pd.DataFrame(q_latent_variable.fetch())
        df_Q_left = df_Q[df_Q['water_port']=='left'][['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_left'})#.reset_index(drop=True)
        df_Q_right = df_Q[df_Q['water_port']=='right'][['session', 'trial', 'action_value']].rename(columns={'action_value': 'Q_right'})#.reset_index(drop=True)
        df_Qs = df_Q_left.merge(df_Q_right)

        sessions_all = np.unique(df_Qs['session'].values)
        sessions = sessions_all[sessions_all > latter_session]


        for session in sessions:
            df_Qs_session = df_Qs[df_Qs['session']==session]
            ## check if Q_left or Q_right are missing
            if np.isnan(df_Qs_session['Q_left'].values).any() or np.isnan(df_Qs_session['Q_right'].values).any():
                Q_left_nan = np.where(np.isnan(df_Qs_session['Q_left'].values))[0]
                Q_right_nan = np.where(np.isnan(df_Qs_session['Q_right'].values))[0]
                print(f' nan present in Qs: region {region}, sess {session}, left {Q_left_nan}, right {Q_right_nan}')
                print(' dropping nan')
                df_Qs_session.dropna(subset=['Q_left', 'Q_right'], inplace=True)

            unit_qc_region_mouse_session = select_unit_qc_region_mouse_session(
                                                all_unit_qc, 
                                                region_annotation=region_ann_lut[region], 
                                                mouse=mouse, 
                                                session=session)
            if len(unit_qc_region_mouse_session) > 0:
                print(f'total number of units in {region} in sess {session}: {len(unit_qc_region_mouse_session)}')
                # fetch keys
                keys2units = gen_keys2units(unit_qc_region_mouse_session)

                trials_q = df_Qs_session['trial'].values
                # get unit activity
                for unit_key in keys2units:
                    unit = unit_key['unit']
                    
                    period_activity = psth_foraging.compute_unit_period_activity(unit_key, period)
                    trials_unit = period_activity['trial']

                    # correct mismatch trials from q and unit
                    trials_q_not_in_unit = trials_q[~np.isin(trials_q, trials_unit)]
                    if np.any(trials_q_not_in_unit):
                        print(f' mismatch: {region} {session}, trials_q_not_in_unit: {trials_q_not_in_unit}')
                        for t in trials_q_not_in_unit:
                            df_Qs.drop(df_Qs[(df_Qs['session']==session) &
                                             (df_Qs['trial']==t)].index, inplace=True)
                    trials_q = trials_q[np.isin(trials_q, trials_unit)]

                    firing_rate = period_activity['firing_rates']
                    # get only the valid trials
                    valid_firing_rate = firing_rate[np.isin(trials_unit, trials_q)]

                    neurons[region].loc[len(neurons[region].index)] = [session, unit, valid_firing_rate]

        q_latents[region] = df_Qs


    # save the data df if not existed
    with open('./neurons_data_match_iti_all.pickle', 'wb') as f_handle:
        pickle.dump(neurons, f_handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./q_latents_match.pickle', 'wb') as handle:
        pickle.dump(q_latents, handle, protocol=pickle.HIGHEST_PROTOCOL)

