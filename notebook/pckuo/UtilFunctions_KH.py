# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:57:31 2022
Functions for Ophys/DataJoint-Behavior Analysis
@author: kenta.hagihara
"""


import datajoint as dj
import numpy as np
from pipeline import lab, get_schema_name, experiment, foraging_model, ephys, foraging_analysis, histology, ccf


def _get_independent_variableKH(unit_key, model_id, var_name=None):
    #modified from _get_unit_independent_variable
    """
    Get independent variable over trial for a specified unit (ignored trials are skipped)
    @param unit_key:
    @param model_id:
    @param var_name
    @return: DataFrame (trial, variables)
    """
    
    #hemi = _get_units_hemisphere(unit_key)
    contra, ipsi = ['right', 'left'] #if hemi == 'left' else ['left', 'right']

    # Get latent variables from model fitting
    q_latent_variable = (foraging_model.FittedSessionModel.TrialLatentVariable
                         & unit_key
                         & {'model_id': model_id})

    # Flatten latent variables to generate columns like 'left_action_value', 'right_choice_prob'
    latent_variables = q_latent_variable.heading.secondary_attributes
    q_latent_variable_all = dj.U('trial') & q_latent_variable
    for lv in latent_variables:
        for prefix, side in zip(['left_', 'right_', 'contra_', 'ipsi_'],
                                ['left', 'right', contra, ipsi]):
            # Better way here?
            q_latent_variable_all *= eval(f"(q_latent_variable & {{'water_port': '{side}'}}).proj({prefix}{lv}='{lv}', {prefix}='water_port')")

    # Add relative and total value
    q_latent_variable_all = q_latent_variable_all.proj(...,
                                                       relative_action_value_lr='right_action_value - left_action_value',
                                                       relative_action_value_ic='contra_action_value - ipsi_action_value',
                                                       total_action_value='contra_action_value + ipsi_action_value')

    # Add choice
    q_independent_variable = (q_latent_variable_all * experiment.WaterPortChoice).proj(...,
                                                                                       choice='water_port',
                                                                                       choice_lr='water_port="right"',
                                                                                       choice_ic=f'water_port="{contra}"')

    # Add reward
    q_independent_variable = (q_independent_variable * experiment.BehaviorTrial.proj('outcome')).proj(...,
                                                                                                       reward='outcome="hit"'
                                                                                                       )

    df = q_independent_variable.fetch(format='frame', order_by='trial').reset_index()
    
    # Compute RPE
    df['rpe'] = np.nan
    df.loc[0, 'rpe'] = df.reward[0]
    for side in ['left', 'right']:
        _idx = df[(df.choice == side) & (df.trial > 1)].index
        df.loc[_idx, 'rpe'] = df.reward.iloc[_idx] - df[f'{side}_action_value'].iloc[_idx - 1].values

    return df if var_name is None else df[['trial', var_name]]


#### For trial alignment
def align_phys_to_behav_trials(phys_barcode, behav_barcode, behav_trialN=None):
    '''
    Align physiology trials (ephys/ophys) to behavioral trials using the barcode
    
    Input: phys_barcode (list), behav_barcode (list), behav_trialN (list)
    Output: a dictionary with fields
        'phys_to_behav_mapping': a list of trial mapping [phys_trialN, behav_trialN]. Use this to trialize events in phys recording
        'phys_not_in_behav': phys trials that are not found in behavioral trials
        'behav_not_in_phys': behavioral trials that are not found in phys trials
        'phys_aligned_blocks': blocks of consecutive phys trials that are aligned with behav
        'behav_aligned_blocks': blocks of consecutive behav trials that are aligned with phys (each block has the same length as phys_aligned_blocks)
        'perfectly_aligned': whether phys and behav trials are perfectly aligned
        
    '''
    
    if behav_trialN is None:
        behav_trialN = np.r_[1:len(behav_barcode) + 1]
    else:
        behav_trialN = np.array(behav_trialN)
        
    behav_barcode = np.array(behav_barcode)
    
    phys_to_behav_mapping = []  # A list of [phys_trial, behav_trial]
    phys_not_in_behav = []  # Happens when the bpod protocol is terminated during a trial (incomplete bpod trial will not be ingested to behavior)
    behav_not_in_phys = []  # Happens when phys recording starts later or stops earlier than the bpod protocol
    behav_aligned_blocks = []  # A list of well-aligned blocks
    phys_aligned_blocks = []  # A list of well-aligned blocks
    bitCollision = [] # Add the trial numbers with the same bitcode for restrospective sanity check purpose (220817KH)
    behav_aligned_last = -999
    phys_aligned_last = -999
    in_a_continous_aligned_block = False # A flag indicating whether the previous phys trial is in a continuous aligned block
                
    for phys_trialN_this, phys_barcode_this in zip(range(1, len(phys_barcode + ['fake']) + 1), phys_barcode + ['fake']):   # Add a fake value to deal with the boundary effect
        behav_trialN_this = behav_trialN[behav_barcode == phys_barcode_this]
        #assert len(behav_trialN_this) <= 1  # Otherwise bitcode must be problematic
        
        #'''
        if len(behav_trialN_this) > 1:
            
            bitCollision.append(behav_trialN_this)
            closest_idx = np.abs(np.array(behav_trialN_this) - phys_trialN_this).argmin()
            behav_trialN_this = behav_trialN_this[closest_idx:closest_idx+1] #only retaining the closest trialN  (220817KH)
        #'''
        if len(behav_trialN_this) == 0 or behav_trialN_this - behav_aligned_last > 1:  # The current continuously aligned block is broken
            # Add a continuously aligned block
            if behav_aligned_last != -999 and phys_aligned_last != -999 and in_a_continous_aligned_block:
                behav_aligned_blocks.append([behav_aligned_block_start_this, behav_aligned_last])
                phys_aligned_blocks.append([phys_aligned_block_start_this, phys_aligned_last])
                
            in_a_continous_aligned_block = False
            
        if len(behav_trialN_this) == 0:
            phys_not_in_behav.append(phys_trialN_this)
        else:
            phys_to_behav_mapping.append([phys_trialN_this, behav_trialN_this[0]])  # The main output
            
            # Cache the last behav-phys matched pair
            behav_aligned_last = behav_trialN_this[0]
            phys_aligned_last = phys_trialN_this
            
            # Cache the start of each continuously aligned block
            if not in_a_continous_aligned_block:  # A new continuous block just starts
                behav_aligned_block_start_this = behav_trialN_this[0]
                phys_aligned_block_start_this = phys_trialN_this
            
            # Switch on the flag
            in_a_continous_aligned_block = True
            
    phys_not_in_behav.pop(-1)  # Remove the last fake value
    behav_not_in_phys = list(np.setdiff1d(behav_trialN, [b for _, b in phys_to_behav_mapping]))
    
    return {'phys_to_behav_mapping': phys_to_behav_mapping,
            'phys_not_in_behav': phys_not_in_behav,
            'behav_not_in_phys': behav_not_in_phys,
            'phys_aligned_blocks': phys_aligned_blocks,
            'behav_aligned_blocks': behav_aligned_blocks,
            'perfectly_aligned': len(phys_not_in_behav + behav_not_in_phys) == 0
            }