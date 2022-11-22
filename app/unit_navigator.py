#%%
import pandas as pd
import streamlit as st
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from PIL import Image
import mpld3
import streamlit.components.v1 as components
from streamlit_plotly_events import plotly_events
import plotly.express as px

from streamlit_util import *
# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
from pipeline.plot import foraging_model_plot

cache_folder = '/Users/han.hou/s3-drive/st_cache/'
cache_fig_folder = '/Users/han.hou/Library/CloudStorage/OneDrive-AllenInstitute/pipeline_report/report/all_units/'  # 
# cache_fig_folder = '/Users/han.hou/s3-drive/all_units/'

st.set_page_config(layout="wide", page_title='Foraging unit navigator')

# always_refetch = st.checkbox('Always refetch', value=False)
ephys_only = st.checkbox('Ephys only', value=True)


@st.experimental_memo
def get_fig_unit_all_in_one(row):
    key = row[0]
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    
    fn = f'*{key["h2o"]}_{sess_date_str}_{key["ins"]}*u{key["unit"]:03}*'
    file = glob.glob(cache_fig_folder + '**/' + fn, recursive=True)[0]
    img = Image.open(file)
    img = img.crop((500, 140, 5400, 3000))
            
    return img

# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }

@st.experimental_memo
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        # with st.spinner(f'Load {table}...'):
            # if not always_refetch and Path(cache_folder + f'{table}.pkl').exists():
            # f'Table {table} read from cache'
        df[table] = pd.read_pickle(cache_folder + f'{table}.pkl')
            # else:
            #     f'Table {table} fetched'
            #     df[table] = table_mapping[table]()
    return df
    

df = load_data(['sessions', 'ephys_units'])


@st.experimental_memo
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   

@st.experimental_memo
def plot_scatter(data, x_name='dQ_iti', y_name='sumQ_iti'):
    fig = px.scatter(data, x=x_name, y=y_name, 
                     color='area_of_interest', symbol="area_of_interest",
                     hover_data=['annotation'])
    
    fig.add_vline(x=2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=-2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=-2.0, line_width=1, line_dash="dash", line_color="black")

    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="black")
    
    fig.update_xaxes(range=[-40, 40])
    fig.update_yaxes(range=[-40, 40])
    
    fig.update_layout(width = 800, height = 800)
    
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    
    return fig

# ------- Layout starts here -------- #

st.markdown('## Foraging sessions')
col1, col2 = st.columns([1, 1.5], gap='small')
with col1:
    if ephys_only:
        selection = aggrid_interactive_table_session(df=df['sessions'].query('ephys_ins > 0'))
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
        

st.markdown('## Unit browser')

col3, col4 = st.columns([1, 1.3], gap='small')
with col3:
    selection_units = aggrid_interactive_table_units(df=df['ephys_units'])
    
with col4:
    if selection_units['selected_rows']:
        unit_fig = get_fig_unit_all_in_one(selection_units['selected_rows'])
        
        st.image(unit_fig, output_format='PNG', width=3000)
        
# %%

with col3:
    # Writes a component similar to st.write()
    fig = plot_scatter(df['ephys_units'])
    
    # Select other Plotly events by specifying kwargs
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                    override_height=800, override_width=800)
    selected_points