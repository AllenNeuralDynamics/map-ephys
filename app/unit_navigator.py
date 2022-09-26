#%%
import pandas as pd
import streamlit as st

from streamlit_util import aggrid_interactive_table
from pipeline import experiment, ephys, lab, psth_foraging
import datajoint as dj 
dj.conn().connect()

#%%

st.title("Hello World")
st.write(
    """
    Hello world
    """
)

#%%
# foraging_sessions = (experiment.Session * lab.WaterRestriction) & (experiment.BehaviorTrial & 'task_protocol = 100')  # Two-lickport foraging
# all_unit_qc = ((ephys.Unit * ephys.ClusterMetric * ephys.UnitStat) & foraging_sessions 
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

df = experiment.Session.fetch(format='frame')
# df = pd.read_csv(R'app\temp.csv')

selection = aggrid_interactive_table(df=df)

if selection:
    st.write("You selected:")
    st.json(selection["selected_rows"])
    dj.conn().connect()