from ctypes import alignment
import os
import logging
import pathlib
import json
from datetime import datetime
from tqdm import tqdm
import re
from itertools import repeat
import pandas as pd
import numpy as np
import datajoint as dj

from .. import get_schema_name
from .. import lab, experiment, ephys, ophys

from pipeline.ingest.utils.paths import get_ophys_sess_dirs


schema = dj.schema(get_schema_name('ingest_ophys'))

log = logging.getLogger(__name__)


TTL_trial_event_length = {'bitcodestart': 20,
                          'go': 1,
                          'reward': 30,
                          'trialend': 40,
                          }
TTL_bar_code_length = 20
TTL_action_event_length = {'choiceL': 2,
                           'choiceR': 3}
TTL_length_tolerance = 0.1  # Allow a 10% error in TTL width

@schema
class OphysIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class OphysDir(dj.Part):
        ''' Path to the ophys directories (could be more than one if the recording has been interrupted) '''
        
        definition = """
        -> master
        ophys_dir:          varchar(255)    # yymmdd
        """

    key_source = (experiment.Session & ['subject_id = 619199']) - ophys.FiberPhotometrySession

    def make(self, key):
        '''
        Ophys .make() function
        '''

        log.info('\n======================================================')
        log.info('OphysIngest().make(): key: {k}'.format(k=key))
           

        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & key).fetch1()
        h2o = sinfo['water_restriction_number']
        log.info(sinfo)

        ophys_dirs = get_ophys_sess_dirs(key)

        if ophys_dirs == []:
            log.info(f'  No ophys folders found. Skipped ........................')
            return
        
        for i, ophys_dir in enumerate(ophys_dirs):  
            # TODO: concatenate across folders when the recording was disrupted.
            # Need to cache raw and ni_time etc. and do insertion after looping over all folders!
            
            log.info(f'  Ophys folder found {i + 1}/{len(ophys_dirs)}: {ophys_dir}')
            
            # --- Ingest meta data
            # Load meta file
            meta_file = list(ophys_dir.glob('meta*'))[0]
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Ingest
            fiber_meta = []
            roi_meta = []
            wavelen_meta = []
            
            # TODO: ingesting ophys.FiberPhotometrySession.Fiber from a meta csv file would be better!
            # Here I just add some fake info in meta.json
            for wavelen in meta['excitation_wavelength']:
                wavelen_meta.append({**key, **wavelen})
            
            for fiber in meta['fiber']:
                fiber_meta.append({**key, **fiber})
            
            for header, (fiber_num, color) in meta['roi'].items():
                roi_meta.append({**key, 'fiber_number': fiber_num, 'header_name': header, 'color': color})
                        
            ophys.FiberPhotometrySession.insert1({**key, 'ni_sample_rate': meta['sample_rate']}, allow_direct_insert=True)
            ophys.FiberPhotometrySession.ExcitationWavelength.insert(wavelen_meta)
            ophys.FiberPhotometrySession.Fiber.insert(fiber_meta)
            ophys.FiberPhotometrySession.ROI.insert(roi_meta)    
            log.info(f'  Meta info inserted!')
                    
            # --- Align timestamp to ni-samples---
            # TODO: please check this part carefully!!
            log.info(f'  Align frame times to ni time...')
            timestamp_file = list(ophys_dir.glob('TimeStamp*'))[0]
            timestamp = pd.read_csv(timestamp_file, names=['bonsai_time', 'frame_count'])
            
            syncpulse_file = list(ophys_dir.glob('TTL_TS*'))[0]
            syncpulse = pd.read_csv(syncpulse_file, names=['bonsai_time'])
            syncpulse_interval_ni_sample = 1 * meta['sample_rate']   # This is hard-coded, assuming syncpulse is 1Hz!
            
            frame_time_ni_sample = []
            
            for ts in timestamp['bonsai_time']:
                nearest_idx = np.argmin(np.abs(ts - syncpulse['bonsai_time']))  # find the nearest sync pulse
                time_diff_bonsai = ts - syncpulse['bonsai_time'][nearest_idx]  # bonsai-time difference between this frame and the nearest sync pulse
                time_diff_ni_sample = np.round(time_diff_bonsai / (1000 / meta['sample_rate']))  # diff in ms / ni_step_size in ms
                
                # This is relative to the first sync pulse! (could be negative if the first frame is before the first sync pulse)
                frame_time_ni_sample.append(nearest_idx * syncpulse_interval_ni_sample + time_diff_ni_sample)
                
            frame_time_ni_ms = np.array(frame_time_ni_sample) * (1000 / meta['sample_rate'])  # ni_sample * ni_step_size in ms
            frame_time_ni_ms = frame_time_ni_ms - frame_time_ni_ms[0]  # So that the time starts from zero (the first frame)
            log.info(f'     Done!')
                        
            # --- Ingest raw data and frame time (in ms, aligned to ni) ---
            # Read raw
            raw_file = list(ophys_dir.glob('Raw*'))[0]
            raw_pd = pd.read_csv(raw_file)
            
            log.info(f'  Ingest raw data...')
            
            for wavelen in wavelen_meta:
                for roi in roi_meta:
                    header_this = roi['header_name']
                    ledstate_this = wavelen['led_state']
                    
                    # raw for this
                    raw_this = raw_pd[raw_pd['LedState'] == ledstate_this][header_this] 
                    # ni_time for this
                    ni_time_this = frame_time_ni_ms[raw_this.index]
                    
                    # do insertion
                    ophys.FiberPhotometrySession.RawTrace.insert1({**key, **roi, **wavelen,
                                                                   'raw': raw_this.to_numpy(),
                                                                   'ni_time': ni_time_this},
                                                                  ignore_extra_fields=True)
        
            log.info(f'     Done!')
           

            # TODO: --- Insert ophys.TrialEvent and ophys.ActionEvent ---
            
            ### Kenta 2022/07/05- added ###
            log.info(f'  Parse TTL (from BPOD recorded with NIDAQ through Bonsai) data...')
            
            TTLsignal_file = list(ophys_dir.glob('TTL_20*'))[0]
            TTLsignal = np.fromfile(TTLsignal_file)
            
            # Sorting NIDAQ-AI channels
            if (len(TTLsignal)/1000) / len(syncpulse) == 1:
                TTLsignal1 = TTLsignal
                
            elif (len(TTLsignal)/1000) / len(syncpulse) == 2:  #this shouldn't happen, though...
                TTLsignal1 = TTLsignal[0::2]
                    
            elif (len(TTLsignal)/1000) / len(syncpulse) >= 3:  # deinterleaved Channel #2 and #3 are raw licks 
                TTLsignal1 = TTLsignal[0::3]
                    
            del TTLsignal 
            
            # analoginputs binalize (TTLsignal1 is the deinterleaved TTL signal from BPOD)
            TTLsignal1[TTLsignal1 < 3] = 0
            TTLsignal1[TTLsignal1 >= 3] = 1
            TTLsignal1_shift = np.roll(TTLsignal1, 1)
            diff = TTLsignal1 - TTLsignal1_shift

            # Sorting
            TTL_p = [] # position(timing) of TTL pulse onset
            TTL_l = [] # length of pulse

            for ii in range(len(TTLsignal1)):
                if diff[ii] == 1:
                    for jj in range(120): #Max length:40
                        if ii+jj > len(TTLsignal1)-1:
                            break
                        
                        if diff[ii+jj] == -1:
                            TTL_p = np.append(TTL_p, ii) 
                            TTL_l = np.append(TTL_l, jj)
                            break
            log.info(f'     Done!')
            
            # 1. Decode barcode and events
            log.info(f'  Decoding trial barcode and events')
            
            # Barcode
            BarcodeP = [p for p, l in zip(TTL_p, TTL_l) if l >= TTL_bar_code_length * (1 - TTL_length_tolerance) 
                                                       and l <= TTL_bar_code_length * (1 + TTL_length_tolerance)]         #Barcode starts with a 20ms pulse
            BarcodeBin = np.zeros((len(BarcodeP), TTL_bar_code_length)) # matrix of 0 or 1, size = (trial#,20)

            for ii in range(len(BarcodeP)):
                for jj in range(20):
                    BarcodeBin[ii,jj] = TTLsignal1[int(BarcodeP[ii] + TTL_bar_code_length * (1.5 + jj) + 5)] # checking the middle of 10ms windows

            ophys_barcode = []  # This will be the list (size = trialN) of 20 char (eg. "01011001001010010100")  

            for bar_code_bin in BarcodeBin:                   
                ophys_barcode.append(''.join([str(int(b)) for b in bar_code_bin]))
                
            # Events
            trial_events = {}
            for trial_event_type, length in TTL_trial_event_length.items():
                trial_events[trial_event_type] = np.array(TTL_p)[(np.array(TTL_l) >= length * (1 - TTL_length_tolerance))
                                                               & (np.array(TTL_l) <= length * (1 + TTL_length_tolerance))] 
                
            action_events = {}
            for action_event_type, length in TTL_action_event_length.items():
                action_events[action_event_type] = np.array(TTL_p)[(np.array(TTL_l) >= length * (1 - TTL_length_tolerance))
                                                                 & (np.array(TTL_l) <= length * (1 + TTL_length_tolerance))] 
                                                                
            log.info(f'     Done!')    
            ### Kenta 2022/07/05- added ###
            # BarChar to be directly compared to behavior
            
            # 2. Align barcode to behavior
            log.info(f'  Align ophys trials to behavior using barcode...') 
            
            # Do trial alignment
            behav_trialN, behav_barcode = (experiment.TrialNote & key & 'trial_note_type = "bitcode"').fetch('trial', 'trial_note', order_by='trial')
            trial_aligned = align_phys_to_behav_trials(ophys_barcode, list(behav_barcode), list(behav_trialN))
            
            if trial_aligned['perfectly_aligned']:
                log.info(f'  perfectly aligned!')
            else:
                log.info(f'    aligned blocks found:\n'
                         f'      ophys: {[f"{s}-{e}" for s,e in trial_aligned["phys_aligned_blocks"]]}\n'
                         f'      behav: {[f"{s}-{e}" for s,e in trial_aligned["behav_aligned_blocks"]]}\n'
                         f'    ophys not in behav: {trial_aligned["phys_not_in_behav"]}\n'
                         f'    behav not in ophys: {trial_aligned["behav_not_in_phys"]}')
                        
            # 3. Insert events
            all_bitcodestart = trial_events['bitcodestart']
            all_bitcodestart = np.append(all_bitcodestart, np.inf)
            all_trialevent_to_insert = []
            all_actionevent_to_insert = []
                                  
            for ophys_trial, behav_trial in trial_aligned['phys_to_behav_mapping']:  # Only iterate over the aligned trials
                # Use bitcodestart to separate trials
                start_this_trial = [ophys_trial - 1]
                start_next_trial = trial_events['bitcodestart'][ophys_trial]
                
                # Add trial events of this trial
                for trial_event_type, times in trial_events.items():
                    pass  # TODO: add items to all_trialevent_to_insert
                
                # Add action events of this trial
                for action_event_type, times in action_events.items():
                    pass  # TODO: add items to all_actionevent_to_insert

            pass  # TODO: insert table ophys.TrialEvent and ophys.ActionEvent
        
            
def align_phys_to_behav_trials(phys_barcode, behav_barcode, behav_trialN=None):
    '''
    Align physiology trials (ephys/ophys) to behavioral trials using the barcode
    
    Input: phys_barcode (list), behav_barcode (list), behav_trialN (list)
    Output: a dictionary with fields
        'phys_to_behav_mapping': a list of trial mapping [phys_trialN, behav_trialN]. Use this to trialize events in phys recording
        'phys_not_in_behav': phys trials that are not found in behavioral trials
        'behav_not_in_phys': behavioral trials that are not found in phys trials
        'phys_aligned_blocks': blocks of consecutive phys trials that are aligned with behav
        'bitCollision': trial numbers with the same bitcode
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
    bitCollision = [] # Add the trial numbers with the same bitcode for restrospective sanity check purpose (220817)
    behav_aligned_last = -999
    phys_aligned_last = -999
    in_a_continous_aligned_block = False # A flag indicating whether the previous phys trial is in a continuous aligned block
                
    for phys_trialN_this, phys_barcode_this in zip(range(1, len(phys_barcode + ['fake']) + 1), phys_barcode + ['fake']):   # Add a fake value to deal with the boundary effect
        behav_trialN_this = behav_trialN[behav_barcode == phys_barcode_this]
        #assert len(behav_trialN_this) <= 1  # Otherwise bitcode must be problematic (collision actually happens often.. 220817KH)
        
        
        if len(behav_trialN_this) > 1:
            
            bitCollision.append(behav_trialN_this)
            closest_idx = np.abs(np.array(behav_trialN_this) - phys_trialN_this).argmin()
            behav_trialN_this = behav_trialN_this[closest_idx:closest_idx+1] #only retaining the closest trialN  (220817KH)
        

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
            'bitCollision': bitCollision,
            'perfectly_aligned': len(phys_not_in_behav + behav_not_in_phys) == 0
            }