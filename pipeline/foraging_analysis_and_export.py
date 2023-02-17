import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt

from . import experiment, ephys, foraging_analysis, foraging_model, util
from . import get_schema_name, create_schema_settings
from .plot import foraging_model_plot
from .model.bandit_model_comparison import BanditModelComparison


"""
A schema doing analysis and export to AWS at the same time.
This allows me to use functions that are more independent of Datajoint and can be readily reused in CodeOcean.
This is not the best way but it is the fastest.

Han, Feb 2023
"""


schema = dj.schema(get_schema_name('foraging_analysis_and_export'), **create_schema_settings)

    
@schema
class SessionLogisticRegression(dj.Computed):
    definition = """
    -> foraging_analysis.SessionTaskProtocol    # Foraging sessions
    trial_group:  varchar(30)    # no_stim_all, ctrl, photostim, photostim_next, photostim_next5
    beta:         varchar(30)    # RewC, UnrC, C, bias
    trials_back:  smallint          
    ---
    mean:     float
    lower_ci:   float
    upper_ci:   float
    """
    
    foraging_sessions = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol in (100, 110, 120)').proj()
    key_source = foraging_sessions & experiment.PhotostimForagingTrial  # Photostim only
    
    def make(self, key):
        
        if_photostim = len((experiment.PhotostimForagingTrial & (experiment.BehaviorTrial & key & 'outcome != "ignore"'))) > 10
        
        if not if_photostim:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            trial_groups = ['all_no_stim']
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 20))
            trial_groups = ['ctrl', 'photostim', 'photostim_next', 'photostim_next5']

            
        # Do logistic regression and generate figures
        start_trial, end_trial = (foraging_analysis.SessionEngagementControl & key).fetch1('start_trial', 'end_trial')
        c, r, _, p, _ = foraging_model.get_session_history(key, remove_ignored=True)
        choice = c[0][start_trial - 1:end_trial]
        reward = np.sum(r, axis=0)[start_trial - 1:end_trial]
        
        if if_photostim:
            non_ignore_trial = (experiment.BehaviorTrial & key & 'outcome != "ignore"').fetch('trial')
            photostim_trial = (experiment.PhotostimForagingTrial & (experiment.BehaviorTrial & key & 'outcome != "ignore"')).fetch('trial')
            photostim_idx = np.nonzero(np.in1d(non_ignore_trial, photostim_trial))[0]   # np.searchsorted(non_ignore_trial, photostim_trial)
        else:
            photostim_idx = None
        
        logistic_regs = foraging_model_plot.plot_session_logistic(choice, reward, photostim_idx=photostim_idx, ax=ax)

        for trial_group, logistic_reg in zip(trial_groups, logistic_regs):
            rows = decode_logistic_reg(logistic_reg)
            self.insert({**key,
                         'trial_group': trial_group,
                         **row} for row in rows)
        # Save figures
        fig.suptitle(util._get_sess_info(key))
            
            
  
def decode_logistic_reg(reg):
    mapper = {'b_RewC': 'RewC', 'b_UnrC': 'UnrC', 'b_C': 'C'}    
    output = []

    for field, name in mapper.items():
        for trial_back in range(reg.b_RewC.shape[1]):
            output.append({'beta': name, 
                           'trials_back': trial_back + 1,
                           'mean': getattr(reg, field)[0, trial_back],
                           'lower_ci': getattr(reg, field + '_CI')[0, trial_back],
                           'upper_ci': getattr(reg, field + '_CI')[1, trial_back]})

    output.append({'beta': 'bias',
                   'trials_back': 0,
                   'mean': reg.bias[0, 0],
                   'lower_ci': reg.bias_CI[0][0],
                   'upper_ci': reg.bias_CI[1][0]})

    return output