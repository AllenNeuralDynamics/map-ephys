'''
from _debug_psth_refactor.ipynb
'''
#%%
import numpy as np
import pandas as pd

import dill

import os; os.chdir(R'/root/capsule/code')
import sys; sys.path.append(R'/root/capsule/code')
from pipeline import lab, experiment, ephys, histology, psth_foraging
import datajoint as dj; dj.conn().connect()

from standalone.to_s3_util import export_df_and_upload


def align_spikes_to_event(spike_time_units, event_times, win):
    """
    Parameters
    -------------
    spike_time_units: list(units)[ndarray(spike times)]
    event_times: dict[align_type: ndarray(event time)]
    win: cut off window, list[start, end]
    
    Returns
    -------------
    spike_time_aligned: dict[align_type: list(unit)[ndarray(trial, aligned_time)]]
    """
    
    for align_type, event_time in event_times.items():
        for spikes in spike_times_units:
            spike_time_aligned = []
            for e_t in event_time:
                s_t = spikes[(e_t + win[0] <= spikes) & (spikes < e_t + win[1])]
                spike_time_aligned.append(s_t - e_t)
        

def session_align_spike_counts(sess_key, align_types=['go_cue'], bin_size=0.02):
    """
    return: spike_count_aligned [unit_qced, trial, time bins]
    """

    q_unit_qc_session = ephys.UnitForagingQC & sess_key & 'unit_minimal_session_qc'
    units = q_unit_qc_session.fetch('KEY')   
    spike_times_units = (ephys.Unit & q_unit_qc_session).fetch('spike_times')
    
    trials, time_bins, aligned_firings = {}, {}, {}
    
    for align_type in align_types:
        q_align_type = psth_foraging.AlignType & {'align_type_name': align_type}

        # -- Get global times for spike and event --
        q_event = ephys.TrialEvent & sess_key & {'trial_event_type': q_align_type.fetch1('trial_event_type')}   # Using ephys.TrialEvent, not experiment.TrialEvent

        # Session-wise event times (relative to session start)
        events, trial_num = q_event.fetch('trial_event_time', 'trial', order_by='trial asc')
        # Make event times also relative to the first sTrig
        events -= ((ephys.TrialEvent & sess_key) & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time')
        events = events.astype(float)

        # Manual correction if necessary (e.g. estimate trialstart from bitcodestart when zaberready is missing)
        events += q_align_type.fetch('time_offset').astype(float)

        # -- Align spike times to each event --
        win = q_align_type.fetch1('psth_win')
        bins = np.arange(win[0], win[1], bin_size)

        # --- Aligned spike count in bins ---
        spike_count_aligned = np.empty([len(units), len(trial_num), len(bins) - 1], dtype='uint8')

        for n, spikes in enumerate(spike_times_units):
            spike_time_aligned = []
            for e_t in events:
                s_t = spikes[(e_t + win[0] <= spikes) & (spikes < e_t + win[1])]
                spike_time_aligned.append(s_t - e_t)
            spike_count_aligned[n, :, :] = np.array(list(map(lambda x: np.histogram(x, bins=bins)[0], spike_time_aligned)))
            
        # --- Save data ---
        trials[align_type] = trial_num
        time_bins[align_type] = np.mean([bins[:-1], bins[1:]], axis=0)
        aligned_firings[align_type] = spike_count_aligned

    aligned_firing = dict(sess_key=sess_key, unit_keys=units, trials=trials, 
                          times=time_bins, align_tos=align_types, aligned_firings=aligned_firings)
    aligned_firing['bin_size'] = bin_size
    
    return aligned_firing


def export_one_session(sess, align_types, bin_size):
    dj.conn().connect()
    session_aligned = session_align_spike_counts(sess, align_types=align_types, bin_size=bin_size)
    fname = f'{sess["subject_id"]}_{sess["session"]}_aligned_spike_counts.pkl'
    
    export_df_and_upload(session_aligned, s3_rel_path='export/', file_name=fname, method='dill_dump')
    # dill.dump(session_aligned, open(export_path + f'{fname}.pkl', 'wb'))
    
    del session_aligned
    print(f'{sess} done')


#%% 
if __name__ == '__main__':

    import multiprocessing as mp
    pool = mp.Pool(8)
    
    session_to_export = (dj.U('subject_id', 'session')
                     & (ephys.UnitForagingQC & 'unit_minimal_session_qc = 1') 
                     & histology.ElectrodeCCFPosition.ElectrodePosition    # With ccf
                     - experiment.PhotostimForagingTrial                   # No photostim
                     ).fetch('KEY')

    align_types = ['go_cue', 'choice', 'iti_start']
    bin_size = 0.01

    # for sess in session_to_export:
    #     export_one_session(sess, align_types, bin_size)

    results = [pool.apply_async(export_one_session, args=(sess, align_types, bin_size)) for sess in session_to_export]
    _ = [r.get() for r in results]
    
    pool.close()
    pool.join()

