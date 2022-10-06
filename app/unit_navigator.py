#%%
import pandas as pd
import streamlit as st
from pathlib import Path

from streamlit_util import aggrid_interactive_table

cache_folder = './app/'
always_refetch = True

#%%

st.set_page_config(page_title='Foraging unit navigator', layout="wide")

"""
Hello world 
"""

#%%
def fetch_ephys_units():
    # df = pd.DataFrame(all_unit_qc.fetch())
    from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
    
    t_iti_delta_Q = (psth_foraging.UnitPeriodLinearFit.Param & 'period in ("iti_all")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
    & 'var_name = "relative_action_value_ic"').proj(t_iti_delta_Q='t', t_abs_iti_delta_Q='ABS(t)')
    
    t_trial_delta_Q = (psth_foraging.UnitPeriodLinearFit.Param & 'period in ("go_to_end")' & 'multi_linear_model = "Q_rel + Q_tot + rpe"' 
    & 'var_name = "relative_action_value_ic"').proj(t_go_to_end_delta_Q='t', t_abs_go_to_end_delta_Q='ABS(t)', _='period')
    
    df_all_unit = pd.DataFrame((
                report.UnitLevelForagingEphysReportAllInOne.key_source * ephys.Unit.proj('unit_uid', 'unit_amp', 'unit_snr')
                * ephys.ClusterMetric.proj('presence_ratio', 'amplitude_cutoff')
                * ephys.UnitStat
                * t_iti_delta_Q * t_trial_delta_Q
                ).fetch())
    
    df_all_unit.to_csv(cache_folder + 'all_unit.csv')
    
    return df_all_unit

def fetch_sessions():
    # with st.spinner(f'Connect to datajoint...'):
    from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis, histology
    import datajoint as dj; dj.conn().connect()
    
    # with st.spinner(f'Fetching...'):
    foraging_sessions = (foraging_analysis.SessionTaskProtocol & 'session_task_protocol=100').proj()
    insertion_numbers = foraging_sessions.aggr(foraging_sessions * ephys.ProbeInsertion, ..., 
                                                    #   keep_all_rows=True, ephys_insertions='IF(COUNT(insertion_number), "yes", "no")')
                                                keep_all_rows=True, ephys_insertions='COUNT(insertion_number)')
    if_histology = foraging_sessions.aggr(foraging_sessions * histology.ElectrodeCCFPosition.ElectrodePosition, ...,
                                          keep_all_rows=True, if_histology='IF(COUNT(ccf_x)>0, "yes", "no")')
    if_photostim_from_behav = foraging_sessions.aggr(foraging_sessions * experiment.PhotostimForagingTrial, ...,
                                          keep_all_rows=True, if_photostim_from_behav='IF(COUNT(trial)>0, "yes", "no")')
    if_photostim_from_ephys = foraging_sessions.aggr(foraging_sessions * (ephys.TrialEvent & 'trial_event_type LIKE "laser%"'), ...,
                                          keep_all_rows=True, if_photostim_from_ephys='IF(COUNT(trial)>0, "yes", "no")')

    df_sessions = pd.DataFrame(((experiment.Session & foraging_sessions)
                                * lab.WaterRestriction.proj('water_restriction_number')
                                * insertion_numbers
                                * if_histology
                                * if_photostim_from_behav
                                * if_photostim_from_ephys)
                               .proj(..., '-rig', '-username', '-session_time')
                               .fetch()
                                )
    # df_sessions['session_date'] = pd.to_datetime(df_sessions['session_date'], format="%Y-%m-%d")
    df_sessions.to_csv(cache_folder + 'sessions.csv')
    
    return df_sessions

# if Path(cache_folder + 'all_unit.csv').exists():
#     df_all_unit = pd.read_csv(cache_folder + 'all_unit.csv')
#     df_sessions = pd.read_csv(cache_folder + 'df_sessions.csv')
# else:
#     'cached all_unit.csv does not exist'
#     df_all_unit = fetch_ephys_units()
#     df_sessions = fetch_sessions()

# h2o = pd.DataFrame(lab.WaterRestriction.proj('water_restriction_number').fetch())
# df = df.merge(h2o, left_on='subject_id', right_on='subject_id')

table_mapping = {
    'sessions': fetch_sessions,
    'ephys_units': fetch_ephys_units,
}

def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        # with st.spinner(f'Load {table}...'):
            if not always_refetch and Path(cache_folder + f'{table}.csv').exists():
                'read from cache'
                df[table] = pd.read_csv(cache_folder + f'{table}.csv')
            else:
                'fetched'
                df[table] = table_mapping[table]()
    return df
    

df = load_data(['sessions', 'ephys_units'])
col1, col2 = st.columns(2, gap='small')

with col1:
    selection = aggrid_interactive_table(df=df['sessions'])
    # selection_units = aggrid_interactive_table(df=df['ephys_units'])
   
with col2:
    if selection:
        st.write("You selected:")
        st.json(selection["selected_rows"])