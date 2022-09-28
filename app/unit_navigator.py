#%%
import pandas as pd
import streamlit as st
from pathlib import Path

from streamlit_util import aggrid_interactive_table

from pipeline import experiment, ephys, lab, psth_foraging, report
import datajoint as dj

#%%

st.title("Foraging unit navigator")

"""
Hello world
"""


#%%
def fetch_tables():
        # foraging_sessions = experiment.Session & (experiment.BehaviorTrial & 'task_protocol = 100')
        # all_unit_qc = (((ephys.Unit.proj('unit_uid', 'unit_amp', 'unit_snr') & foraging_sessions) *
        #                 ephys.ClusterMetric.proj('presence_ratio', 'amplitude_cutoff') * 
        #                 ephys.UnitStat)                 
        #                 & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.1' & 'isi_violation < 0.5' & 'unit_amp > 70')

        # t_significant_trial = (ephys.Unit & ((psth_foraging.UnitPeriodLinearFit * 
        #                                 psth_foraging.UnitPeriodLinearFit.Param
        #                                 & 'period in ("go_to_end")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
        #                                 & 'var_name = "relative_action_value_ic"' 
        #                                 & 'ABS(t) > 2'))).proj()

        # t_significant_iti = (ephys.Unit & ((psth_foraging.UnitPeriodLinearFit * 
        #                                 psth_foraging.UnitPeriodLinearFit.Param
        #                                 & 'period in ("iti_all")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
        #                                 & 'var_name = "relative_action_value_ic"' 
        #                                 & 'ABS(t) > 2'))).proj()   

        # t_sign = t_significant_iti + t_significant_iti
        # key_source = t_sign & (ephys.Unit & foraging_model.FittedSessionModel & all_unit_qc.proj()).proj()

        # foraging_sessions = (experiment.Session * lab.WaterRestriction) & (experiment.BehaviorTrial & 'task_protocol = 100')
        # all_unit_qc = ((ephys.Unit.proj('unit_uid', 'unit_amp', 'unit_snr') *
        #                 ephys.ClusterMetric.proj('presence_ratio', 'amplitude_cutoff') * 
        #                 ephys.UnitStat * lab.WaterRestriction.proj('water_restriction_number')) 
        #                 & foraging_sessions 
        #                 & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.1' & 'isi_violation < 0.5' & 'unit_amp > 70')
        # q_unit = (all_unit_qc * psth_foraging.UnitPeriodLinearFit * psth_foraging.UnitPeriodLinearFit.Param)

    # df = pd.DataFrame(all_unit_qc.fetch())
    t_iti_delta_Q = (psth_foraging.UnitPeriodLinearFit.Param & 'period in ("iti_all")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
    & 'var_name = "relative_action_value_ic"').proj(t_iti_delta_Q='t', t_abs_iti_delta_Q='ABS(t)')
    
    t_trial_delta_Q = (psth_foraging.UnitPeriodLinearFit.Param & 'period in ("go_to_end")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
    & 'var_name = "relative_action_value_ic"').proj(t_go_to_end_delta_Q='t', t_abs_go_to_end_delta_Q='ABS(t)', _='period')
    
    df_all_unit = pd.DataFrame(
                report.UnitLevelForagingEphysReportAllInOne.key_source * ephys.Unit.proj('unit_uid', 'unit_amp', 'unit_snr')
                * ephys.ClusterMetric.proj('presence_ratio', 'amplitude_cutoff')
                * ephys.UnitStat
                * t_iti_delta_Q * t_trial_delta_Q
                ).fetch()
    df_all_unit.to_csv('/Users/han.hou/Downloads/all_unit.csv')
    
    return df_all_unit

if Path('/Users/han.hou/Downloads/all_unit.csv').exists():
    df_all_unit = pd.read_csv('/Users/han.hou/Downloads/all_unit.csv')
else:
    'cached all_unit.csv does not exist'
    df_all_unit = fetch_tables()

# h2o = pd.DataFrame(lab.WaterRestriction.proj('water_restriction_number').fetch())
# df = df.merge(h2o, left_on='subject_id', right_on='subject_id')

col1, col2 = st.columns(2, gap='small')

with col1:
    selection = aggrid_interactive_table(df=df_all_unit)
    
with col2:
    if selection:
        st.write("You selected:")
        st.json(selection["selected_rows"])