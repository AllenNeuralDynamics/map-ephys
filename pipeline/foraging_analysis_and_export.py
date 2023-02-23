import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from . import experiment, ephys, foraging_analysis, foraging_model, util, report
from . import get_schema_name, create_schema_settings
from .plot import foraging_model_plot, foraging_plot
from .model.bandit_model_comparison import BanditModelComparison


"""
A schema doing analysis and export to AWS at the same time.
This allows me to use functions that are more independent of Datajoint and can be readily reused in CodeOcean.
This is not the best way but it is the fastest.

Han, Feb 2023
"""


schema = dj.schema(get_schema_name('foraging_analysis_and_export'), **create_schema_settings)

report_cfg = dj.config['stores']['report_store']
store_stage = pathlib.Path(report_cfg['stage'])
    
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
    key_source = foraging_sessions  # & experiment.PhotostimForagingTrial  # Photostim only
    
    def make(self, key):
        
        if_photostim = len((experiment.PhotostimForagingTrial & (experiment.BehaviorTrial & key & 'outcome != "ignore"'))) > 10
        
        if not if_photostim:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
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
        water_res_num, sess_date = report.get_wr_sessdatetime(key)
        sess_dir = store_stage / 'all_sessions' / 'logistic_regression' / water_res_num
        sess_dir.mkdir(parents=True, exist_ok=True)
        
        fn_prefix = f'{water_res_num}_{sess_date.split("_")[0]}_{key["session"]}_'

        fig.suptitle(util._get_sess_info(key))
            
        fig_dict = report.save_figs(
            (fig,),
            ('logistic_regression',),
            sess_dir, fn_prefix)
        
        SessionLogisticRegressionReport.insert1({**key, **fig_dict}, 
                                                ignore_extra_fields=True,
                                                allow_direct_insert=True)

@schema
class SessionLogisticRegressionReport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    logistic_regression: filepath@report_store
    """    
    
    def make(self, key):
        """
        to remove the figures: 
        1. foraging_analysis_and_export.SessionLogisticRegressionReport.delete()
        2. (foraging_analysis_and_export.schema.external['report_store'] & 'filepath LIKE "%logistic%"').delete(delete_external_files=True)
        """
        pass
    

@schema
class SessionBehaviorFittedChoiceReport(dj.Computed):
    definition = """
    -> foraging_analysis.SessionTaskProtocol    # Foraging sessions
    model_id:   int  # specific ids and -1 (best for each session)
    ---
    fitted_choice: filepath@report_store
    """
    
    key_source = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol in (100, 110, 120)')
    
    def make(self, key):
        model_to_plot = [None, 
                         8,   # learning rate, e-greedy
                        11,   # tau1, tau2, softmax
                        14,   # Hattori2019, alpha_Rew, alpha_Unr, delta, softmax
                        15,   # 8 + ck
                        17,   # 11 + ck
                        20,   # Hattori 2019 + CK
                        21,   # Hattori 2019 + CK one trial
                         ]
        
        for model_id in model_to_plot:
            # generate figure
            fig, _, model_plotted = foraging_model_plot.plot_session_fitted_choice(sess_key=key,
                                                                                    specified_model_ids=model_id,
                                                                                    model_comparison_idx=0, 
                                                                                    sort='aic',
                                                                                    first_n=1, last_n=0,
                                                                                    remove_ignored=False, 
                                                                                    smooth_factor=5,
                                                                                    ax=None,
                                                                                    vertical=False)
            if len(model_plotted):
                model_id_plotted = model_plotted.iloc[0].model_id
                model_str = f'model_{"best_" if model_id is None else ""}{model_id_plotted}'
            else:
                model_str = 'model_None'
            
            # Save figures
            water_res_num, sess_date = report.get_wr_sessdatetime(key)
            sess_dir = store_stage / 'all_sessions' / 'fitted_choice' / water_res_num
            sess_dir.mkdir(parents=True, exist_ok=True)
            
            fn_prefix = f'{water_res_num}_{sess_date.split("_")[0]}_{key["session"]}_{model_str}_' \
                        
                
            fig_dict = report.save_figs(
                (fig,),
                ('fitted_choice',),
                sess_dir, fn_prefix)
            
            plt.close('all')

            self.insert1({**key, **fig_dict, 
                          'model_id': -1 if model_id is None else model_id}, 
                        ignore_extra_fields=True,
                        allow_direct_insert=True)



@schema
class SessionLickPSTHReport(dj.Computed):
    definition = """
    -> foraging_analysis.SessionTaskProtocol    # Foraging sessions
    ---
    lick_psth: filepath@report_store
    """
    
    key_source = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol in (100, 110, 120)')
    
    def make(self, key):
        # generate figure
        axs = foraging_plot.plot_lick_psth(key)
        fig = axs[-1].get_figure()
        
        # save figures
        water_res_num, sess_date = report.get_wr_sessdatetime(key)
        sess_dir = store_stage / 'all_sessions' / 'lick_psth' / water_res_num
        sess_dir.mkdir(parents=True, exist_ok=True)
        
        fn_prefix = f'{water_res_num}_{sess_date.split("_")[0]}_{key["session"]}_' \
            
        fig_dict = report.save_figs(
            (fig,),
            ('lick_psth',),
            sess_dir, fn_prefix)
        
        plt.close('all')

        self.insert1({**key, **fig_dict}, 
                    ignore_extra_fields=True,
                    allow_direct_insert=True)
    
    
# -------- Helpers ----------
    
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