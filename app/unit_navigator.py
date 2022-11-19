#%%
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import streamlit.components.v1 as components

from streamlit_util import *
# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
from pipeline.plot import foraging_model_plot

cache_folder = './app/'

st.set_page_config(layout="wide", page_title='Foraging unit navigator')

always_refetch = st.checkbox('Always refetch', value=False)
ephys_only = st.checkbox('Ephys only', value=True)



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
    
    df_all_unit.to_pickle(cache_folder + 'ephys_units.pkl')
    
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
    df_sessions.to_pickle(cache_folder + 'sessions.pkl')
    
    return df_sessions


def get_fig_unit_all_in_one():
    
    return fig

table_mapping = {
    'sessions': fetch_sessions,
    'ephys_units': fetch_ephys_units,
}

# @st.experimental_memo
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        # with st.spinner(f'Load {table}...'):
            if not always_refetch and Path(cache_folder + f'{table}.pkl').exists():
                f'Table {table} read from cache'
                df[table] = pd.read_pickle(cache_folder + f'{table}.pkl')
            else:
                f'Table {table} fetched'
                df[table] = table_mapping[table]()
    return df
    

df = load_data(['sessions', 'ephys_units'])


@st.experimental_memo
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   






col1, col2 = st.columns([1, 2], gap='small')
with col1:
    if ephys_only:
        selection = aggrid_interactive_table_session(df=df['sessions'].query('ephys_insertions > 0'))
    else:
        selection = aggrid_interactive_table_session(df=df['sessions'])
        # selection_units = aggrid_interactive_table(df=df['ephys_units'])

with col2:
    if selection["selected_rows"]:
        # st.write("You selected:")
        # st.json(selection["selected_rows"])
        fig = get_fig(selection["selected_rows"])
        
        # fig_html = mpld3.fig_to_html(fig)
        # components.html(fig_html, height=600)
        st.write(fig)
        
col3, col4 = st.columns([1, 3], gap='small')
with col3:
    selection_units = aggrid_interactive_table_units(df=df['ephys_units'])
    st.write(selection_units)
    
# %%
