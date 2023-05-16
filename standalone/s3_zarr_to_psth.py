#%%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import itertools

from pynwb import NWBFile, TimeSeries, NWBHDF5IO
import xarray as xr
import zarr
import dill
import scipy

import s3fs
from to_s3_util import export_df_and_upload


nwb_folder = '/root/capsule/data/s3/export/nwb/'
zarr_folder = '/root/capsule/data/s3/export/psth/'


session_keys =  dill.load(open(zarr_folder + 'session_keys.pkl', 'rb'))


def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)


# --- Get PSTH from zarr store on S3 ---
fs = s3fs.S3FileSystem(anon=False)
s3_zarr_root = f's3://aind-behavior-data/Han/ephys/export/psth'

unit_key = {'subject_id': 482353, 'session': 36, 'insertion_number': 2, 'unit': 362}

f_zarr = fs.glob(f'{s3_zarr_root}/{unit_key["subject_id"]}_{unit_key["session"]}*.zarr')

ds = xr.open_zarr(f's3://{f_zarr[0]}', consolidated=True)
df_unit_key = pd.DataFrame.from_dict(ds.attrs['unit_keys'])
unit_ind = df_unit_key.query(f'insertion_number == {unit_key["insertion_number"]} & unit == {unit_key["unit"]}').index

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 1, figsize=(5, 5))
# ds.spike_count_aligned_to_go_cue[unit_ind, :, :].plot(ax=axes)

# st.pyplot(fig)

#%%
# Define PSTHs grouped by
smooth_sigma = 0.1  # 100 ms
psth_grouped_by = {
                    'choice_and_reward':
                        {
                        'prod': {'choice_lr': [0, 1], 
                                    'reward': [0, 1]},
                        'name': {'choice_lr': ['left', 'right'],
                                    'reward': ['no_rew', 'rew']},
                        'addition': {},
                        'plot_spec': {'choice_lr': {'marker_color': ['#FF6767', '#677EFF']}, 
                                        'reward': {'line_dash': ['solid', 'dot'], 'line_width': [2, 1]}}
                        },
                    
                    'dQ_quantile_5':
                        {
                        'prod': {'relative_action_value_lr': 'quantile_5',
                                },
                        'name': {'relative_action_value_lr': [f'dQ_quantile_{n + 1}' for n in range(5)],
                                },
                        'addition': {},
                        'plot_spec': {'relative_action_value_lr': {'marker_color': [f'rgba(0, 0, 0, {t})' for t in (0.1, 0.3, 0.5, 0.7, 0.9)]}, 
                                     }
                        }
                    }

#%%
for group_method in psth_grouped_by:
    psth_grouped_by[group_method]['trial_select'] = {}
    psth_grouped_by[group_method]['plot_setting'] = {}
    
    # Generate trial selection and plot setting
    for var, cats in psth_grouped_by[group_method]['prod'].items():
        if 'quantile' in cats:  # Grouped by quantiles of a continuous variable
            q_n = int(cats.split("_")[-1])
            q_cut = pd.qcut(ds[var].values, q=q_n, labels=False, duplicates='drop')  # [0, 1, 2, ..., q_cut-1]
            psth_grouped_by[group_method]['trial_select'][var] = [q_cut == q for q in range(q_n)]
        else:  # Grouped by discrete categories
            psth_grouped_by[group_method]['trial_select'][var] = [ds[var] == cat for cat in cats]
        
        psth_grouped_by[group_method]['plot_setting'][var] = []
        for ind in range(len(psth_grouped_by[group_method]['trial_select'][var])):
            psth_grouped_by[group_method]['plot_setting'][var].append({key: values[ind] for key, values in psth_grouped_by[group_method]['plot_spec'][var].items()})
            
    prod_trials = list(itertools.product(*psth_grouped_by[group_method]['trial_select'].values()))
    grouped_trials = [np.all(prod_trial, axis=0) for prod_trial in prod_trials]
    
    prod_names = list(itertools.product(*psth_grouped_by[group_method]['name'].values()))
    grouped_names = pd.Categorical([', '.join(prod_name) for prod_name in prod_names], ordered=True)
    
    prod_settings = list(itertools.product(*psth_grouped_by[group_method]['plot_setting'].values()))
    grouped_settings = [{key: values for setting in prod_setting for key, values in setting.items()} for prod_setting in prod_settings]
    ds[f'{group_method}_plot_settings'] = xr.DataArray(grouped_settings, dims=[group_method])  # Save plot settings to dataset
    
    # Add trial condition column
    ds[f'grouped_by_{group_method}'] = xr.DataArray(np.full(ds.trial.shape, np.nan).astype(pd.Categorical), dims=['trial'])
    for i, select_trials in enumerate(grouped_trials):
        ds[f'grouped_by_{group_method}'][select_trials] = grouped_names[i]
        
    # For each align to
    for align_to in ds.align_tos:
        # Convert the unit_period_spike_counts data variable to a DataArray
        da_this = ds[f'spike_count_aligned_to_{align_to}'].copy()
        # Add the categorical condition variable to the DataArray (to not groupby the whole dataset)
        trial_groups = ds[f'grouped_by_{group_method}']
        if 'choice' in align_to:
            trial_groups = trial_groups[~trial_groups.isnull()]
            
        trial_dim = 'trial' if 'choice' not in align_to else 'finished_trial'
        da_this[f'{group_method}'] = ((trial_dim), trial_groups.data)

        # Group the unit_period_spike_counts DataArray by condition_ordered along the trial dimension
        da_this.data = halfgaussian_filter1d(da_this.astype(float), sigma=smooth_sigma/ds.attrs['bin_size'], axis=-1) / ds.attrs['bin_size'] # Smoothed, in spikes/s
        mean = da_this.groupby(f'{group_method}').mean()
        sem = da_this.groupby(f'{group_method}').std() / np.sqrt(da_this[trial_dim].groupby(f'{group_method}').count())
        ds[f'psth_aligned_to_{align_to}_grouped_by_{group_method}'] = xr.concat([mean, sem], dim=pd.Index(['mean', 'sem'], name="stat"))
        
ds.attrs['psth_grouped_by'] = psth_grouped_by
        

# %%
## ======= Plotting =======
from plotly_util import add_plotly_errorbar
from plotly.subplots import make_subplots

# --- Plot PSTH ---
# group_by = 'choice_and_reward'
group_by = 'dQ_quantile_5'

fig = make_subplots(rows=1, cols=len(ds.align_tos), subplot_titles=ds.align_tos,
                    shared_yaxes=True)

for col, align_to in enumerate(ds.align_tos):
    for condition in ds[group_by].data:
        t = ds[f't_to_{align_to}'].data
        mean = ds[f'psth_aligned_to_{align_to}_grouped_by_{group_by}'].sel(stat='mean', unit=unit_ind, **{group_by: condition}, drop=True)[0, :].values
        sem = ds[f'psth_aligned_to_{align_to}_grouped_by_{group_by}'].sel(stat='sem', unit=unit_ind, **{group_by: condition}, drop=True)[0, :].values
        add_plotly_errorbar(x=t, y=mean, err=sem, name=condition,   
                            fig=fig, subplot_kwargs=dict(row=1, col=col + 1), 
                            alpha=0.1, mode="lines", showlegend=True if col == 0 else False,
                            **ds[f'{group_by}_plot_settings'].sel(**{group_by: condition}).item())
    

fig.update_layout(
        font=dict(size=17),
        width=1000, height=400,
        xaxis_title=f'Time (s)',
        yaxis_title='PSTH (spikes/s)',
        # title=f'{[meta[x] for x in ["align_to", "time_win"]]}',
        hovermode='closest',
            )

# plotly_events(fig, click_event=False, hover_event=False, select_event=False, override_height=fig.layout.height, override_width=fig.layout.width)
# %%
