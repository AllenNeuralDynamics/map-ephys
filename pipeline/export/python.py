import datajoint as dj
import numpy as np
import pathlib
from tqdm import tqdm

from datetime import datetime
from pipeline import lab, experiment

        
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
    

    # %%
    foraging_sessions = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol = 100') * lab.WaterRestriction  # Two-lickport foraging
    
    all_foraging_subject = (dj.U('water_restriction_number') & foraging_sessions).fetch('KEY')
    for subject_key in all_foraging_subject:
        sessions_this_subject = (foraging_sessions & subject_key).fetch('KEY')
        h2o = (lab.WaterRestriction & subject_key).fetch1('water_restriction_number')
        
        # Skip if npy already exists
        if pathlib.Path(f'./report/behavior/{h2o}.npy').exists():
            print(f'Skip {h2o} because .npy already exists...')
            continue        
        
        pathlib.Path(f'./report/behavior/{h2o}/').mkdir(parents=True, exist_ok=True)
        this_subject = dict()
            
        for session_key in sessions_this_subject:
            choice_history, reward_history, _ , p_reward, _ = get_session_history(session_key, remove_ignored=False)
            try:
                trial_num, foraging_efficiency = (foraging_analysis.SessionStats & session_key).fetch1('session_total_trial_num', 'session_foraging_eff_optimal')
            except:
                print(f'Error in fetching foraging_efficiency for {h2o}, {session_key}!!')
                continue
            
            fig, ax = foraging_model_plot.plot_session_lightweight([choice_history, reward_history, p_reward])  # Include ignored trials
            # foraging_model_plot.plot_session_fitted_choice(session_key)    
            ax.text(0, 1.1, util._get_sess_info(session_key), fontsize=10, transform=ax.transAxes)
            fig.savefig(f'./report/behavior/{h2o}/{h2o}_Session_{session_key["session"]:02}')
            
            this_subject[session_key["session"]] = {'choice_history': choice_history,
                                                    'reward_history': reward_history,
                                                    'p_reward': p_reward,
                                                    'trial_num': trial_num,
                                                    'foraging_efficiency': foraging_efficiency}
            plt.close()
            print(f'Done {h2o}, {session_key}')
           
        np.save(f'./report/behavior/{h2o}.npy', this_subject, allow_pickle=True)
        