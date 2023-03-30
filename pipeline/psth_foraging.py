import logging
from functools import partial
from inspect import getmembers
import numpy as np
import datajoint as dj
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from . import (lab, experiment, ephys)
[lab, experiment, ephys]  # NOQA

from . import get_schema_name, dict_to_hash, create_schema_settings
from pipeline import foraging_model, foraging_analysis, histology
from pipeline.util import _get_unit_independent_variable, _get_units_hemisphere



schema = dj.schema(get_schema_name('psth_foraging'), **create_schema_settings)
log = logging.getLogger(__name__)


@schema
class TrialCondition(dj.Lookup):
    '''
    TrialCondition: Manually curated condition queries.

    Used to define sets of trials which can then be keyed on for downstream
    computations.
    '''

    definition = """
    trial_condition_name:       varchar(128)     # user-friendly name of condition
    ---
    trial_condition_hash:       varchar(32)     # trial condition hash - hash of func and arg
    unique index (trial_condition_hash)
    trial_condition_func:       varchar(36)     # trial retrieval function
    trial_condition_arg:        longblob        # trial retrieval arguments
    """

    @property
    def contents(self):
        contents_data = [
                     
            # ----- Foraging task -------
            {
                'trial_condition_name': 'L_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'L_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },            {
                'trial_condition_name': 'L_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'L_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'miss',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'miss',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'LR_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'L_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right', 
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },           
            {
                'trial_condition_name': 'ignore',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    # 'water_port': 'right', 
                    # 'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            }
            
        ]

        # PHOTOSTIM conditions. Not implemented for now 
        
        # stim_locs = [('left', 'alm'), ('right', 'alm'), ('both', 'alm')]
        # for hemi, brain_area in stim_locs:
        #     for instruction in (None, 'left', 'right'):
        #         condition = {'trial_condition_name': '_'.join(filter(None, ['all', 'noearlylick',
        #                                                                     '_'.join([hemi, brain_area]), 'stim',
        #                                                                     instruction])),
        #                      'trial_condition_func': '_get_trials_include_stim',
        #                      'trial_condition_arg': {
        #                          **{'_outcome': 'ignore',
        #                             'task': 'audio delay',
        #                             'task_protocol': 1,
        #                             'early_lick': 'no early',
        #                             'auto_water': 0,
        #                             'free_water': 0,
        #                             'stim_laterality': hemi,
        #                             'stim_brain_area': brain_area},
        #                          **({'trial_instruction': instruction} if instruction else {})}
        #                      }
        #         contents_data.append(condition)

        return ({**d, 'trial_condition_hash':
            dict_to_hash({'trial_condition_func': d['trial_condition_func'],
                          **d['trial_condition_arg']})}
                for d in contents_data)

    @classmethod
    def get_trials(cls, trial_condition_name, trial_offset=0):
        return cls.get_func({'trial_condition_name': trial_condition_name}, trial_offset)()

    @classmethod
    def get_cond_name_from_keywords(cls, keywords):
        matched_cond_names = []
        for cond_name in cls.fetch('trial_condition_name'):
            match = True
            tmp_cond = cond_name
            for k in keywords:
                if k in tmp_cond:
                    tmp_cond = tmp_cond.replace(k, '')
                else:
                    match = False
                    break
            if match:
                matched_cond_names.append(cond_name)
        return sorted(matched_cond_names)

    @classmethod
    def get_func(cls, key, trial_offset=0):
        self = cls()

        func, args = (self & key).fetch1(
            'trial_condition_func', 'trial_condition_arg')

        return partial(dict(getmembers(cls))[func], trial_offset, **args)

    @classmethod
    def _get_trials_exclude_stim(cls, trial_offset, **kwargs):
        # Note: inclusion (attr) is AND - exclusion (_attr) is OR
        log.debug('_get_trials_exclude_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set((experiment.Photostim * experiment.PhotostimBrainRegion
                          * experiment.PhotostimEvent).heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set((experiment.BehaviorTrial * experiment.WaterPortChoice).heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}
        
        q = (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) -
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())
        
        if trial_offset:
            return experiment.BehaviorTrial & q.proj(_='trial', trial=f'trial + {trial_offset}')
        else:
            return q

    @classmethod
    def _get_trials_include_stim(cls, trial_offset, **kwargs):
        # Note: inclusion (attr) is AND - exclusion (_attr) is OR
        log.debug('_get_trials_include_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set((experiment.Photostim * experiment.PhotostimBrainRegion
                          * experiment.PhotostimEvent).heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set((experiment.BehaviorTrial * experiment.WaterPortChoice).heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        q = (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) &
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())
    
        if trial_offset:
            return experiment.BehaviorTrial & q.proj(_='trial', trial=f'trial + {trial_offset}')
        else:
            return q

    
@schema 
class AlignType(dj.Lookup):
    """
    Define flexible psth alignment types for the foraging task.

    For the previous delay-response task, we only align spikes to the go cue, and all PSTH has been aligned during ephys
    ingestion (see `ephys.Unit.TrialSpikes`).

    Here, for the foraging task, we would like to align spikes to any event, such as go cue, choice (reward), ITI, and
    even the next trial's trial start. Because of this flexibility, we chose not to do alignment during ingestion, but
    use this table to define all possible alignment types we want, and compute the PSTHs on-the-fly.

    Notes:
    1. When we fetch event times for `experiment.TrialEventType`, we use NI time (`ephys.TrialEvent`) instead of bpod
       time (`experiment.TrialEvent`). See `compute_unit_psth_and_raster()`. See also
       `ingest.util.compare_ni_and_bpod_times()` for a comparison between NI and bpod time and why NI time is more
       reliable.

    2. `trial_offset` is used to specify alignment type like the trial start of the *next* trial. This is
       implemented by adding a `trial_offset` in the method `TrialCondition.get_trials()` above.

       Say, we want to compute PSTH conditioned on all rewarded trials and aligned to the next trial start. Assuming
       trials #1, 5, and 10 are the rewarded trials, then with `trial_offset` = 1, `TrialCondition.get_trials()` will
       return trials #2, 6, and 11. Using this shifted `trial_keys`, `compute_unit_psth_and_raster` will effectively
       align the spikes to the *next* trial start of all rewarded trials.

    """
    
    definition = """
    align_type_name: varchar(32)   # user-friendly name of alignment type
    ---
    -> experiment.TrialEventType
    align_type_description='':    varchar(256)    # description of this align type
    trial_offset=0:      smallint         # e.g., offset = 1 means the psth will be aligned to the event of the *next* trial.
    time_offset=0:       Decimal(10, 5)   # will be added to the event time for manual correction (e.g., bitcodestart to actual zaberready)  
    psth_win:            tinyblob    # [t_min, t_max], time window by which `compute_unit_psth_and_raster` counts spikes
    xlim:                tinyblob    # [x_min, x_max], default xlim for plotting PSTH (could be overridden during plotting)
    """
    contents = [
        ['trial_start', 'zaberready', '', 0, 0, [-3, 2], [-2, 1]],
        ['go_cue', 'go', '', 0, 0, [-2, 5], [-1, 3]],
        ['first_lick_after_go_cue', 'choice', 'first non-early lick', 0, 0, [-2, 5], [-1, 3]],
        ['choice', 'choice', 'first non-early lick', 0, 0, [-2, 5], [-1, 3]],  # Alias for first_lick_after_go_cue
        ['iti_start', 'trialend', '', 0, 0, [-3, 10], [-3, 5]],
        ['next_trial_start', 'zaberready', '', 1, 0, [-10, 3], [-8, 1]],
        ['next_two_trial_start', 'zaberready', '', 2, 0, [-10, 5], [-8, 3]],

        # In the first few sessions, zaber moter feedback is not recorded,
        # so we don't know the exact time of trial start ('zaberready').
        # We have to estimate actual trial start by
        #   bitcodestart + bitcode width (42 ms for first few sessions) + zaber movement duration (~ 104 ms, very repeatable)
        ['trial_start_bitcode', 'bitcodestart', 'estimate actual trial start by bitcodestart + 146 ms', 0, 0.146, [-3, 2], [-2, 1]],
        ['next_trial_start_bitcode', 'bitcodestart', 'estimate actual trial start by bitcodestart + 146 ms', 1, 0.146, [-10, 3], [-8, 1]],
        ['next_two_trial_start_bitcode', 'bitcodestart', '', 2, 0.146, [-10, 5], [-8, 3]],
    ]


@schema
class IndependentVariable(dj.Lookup):
    """
    Define independent variables over trial to generate psth or design matrix of regression
    """
    definition = """
    var_name:  varchar(50)
    ---
    desc:   varchar(200)
    """

    @property
    def contents(self):
        contents = [
            # Model-independent (trial facts)
            ['choice_lr', 'left (0) or right (1)'],
            ['choice_ic', 'ipsi (0) or contra (1)'],
            ['choice_ic_next', 'ipsi (0) or contra (1), next choice'],  # Next choice
            ['reward', 'miss (0) or hit (1)'],
            
            # Model-dependent (latent variables)
            ['relative_action_value_lr', 'relative action value (Q_r - Q_l)'],
            ['relative_action_value_ic', 'relative action value (Q_contra - Q_ipsi)'],
            ['total_action_value', 'total action value (Q_r + Q_l)'],
            ['rpe', 'outcome - Q_chosen'],
            
            # Autoregression and linear trend
            ['trial_normalized', 'trial number normalized to [0, 1]'],
        ]

        latent_vars = foraging_model.FittedSessionModel.TrialLatentVariable.heading.secondary_attributes

        for side in ['left', 'right', 'ipsi', 'contra']:
            for var in (latent_vars):
                contents.append([f'{side}_{var}', f'{side} {var}'])
        
        # auto regression
        for shift in range(1, 11):
            contents.append([f'firing_{shift}_back', f'firing rate of {shift} trials back'])

        return contents


@schema
class LinearModel(dj.Lookup):
    """
    Define multivariate linear models for PeriodSelectivity fitting
    """
    definition = """
    multi_linear_model: varchar(100)
    ---
    if_intercept: tinyint   # Whether intercept is included
    """

    class X(dj.Part):
        definition = """
        -> master
        -> IndependentVariable
        """

    @classmethod
    def load(cls):
        contents = [
            # Keep the names for backward-compatibility
            ['Q_l + Q_r + rpe', 1, ['left_action_value', 'right_action_value', 'rpe']],
            ['Q_c + Q_i + rpe', 1, ['contra_action_value', 'ipsi_action_value', 'rpe']],
            ['Q_rel + Q_tot + rpe', 1, ['relative_action_value_ic', 'total_action_value', 'rpe']],  
            
            ['dQ, sumQ, rpe, C*2', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                       'choice_ic', 'choice_ic_next']],
            ['dQ, sumQ, rpe, C*2, t', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                          'choice_ic', 'choice_ic_next', 'trial_normalized']],
            ['dQ, sumQ, rpe, C*2, R*1', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                            'choice_ic', 'choice_ic_next', *[f'firing_{shift}_back' for shift in range(1, 2)]]],
            ['dQ, sumQ, rpe, C*2, R*1, t', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                               'choice_ic', 'choice_ic_next', *[f'firing_{shift}_back' for shift in range(1, 2)], 'trial_normalized']],
            ['dQ, sumQ, rpe, C*2, R*5, t', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                               'choice_ic', 'choice_ic_next', *[f'firing_{shift}_back' for shift in range(1, 6)], 'trial_normalized']],
            ['dQ, sumQ, rpe, C*2, R*10, t', 1, ['relative_action_value_ic', 'total_action_value', 'rpe', 
                                               'choice_ic', 'choice_ic_next', *[f'firing_{shift}_back' for shift in range(1, 11)], 'trial_normalized']],
            ['contraQ, ipsiQ, rpe, C*2, R*5, t', 1, ['contra_action_value', 'ipsi_action_value', 'rpe', 
                                               'choice_ic', 'choice_ic_next', *[f'firing_{shift}_back' for shift in range(1, 6)], 'trial_normalized']],
        ]

        for m in contents:
            cls.insert1(m[:2], skip_duplicates=True)
            for iv in m[2]:
                cls.X.insert1([m[0], iv], skip_duplicates=True)


@schema
class LinearModelPeriodToFit(dj.Lookup):
    """
    Subset of experiment.PeriodForaging (or adding sliding window here in the future?)
    """
    definition = """
    -> experiment.PeriodForaging
    """

    contents = [
        ('before_2', ), ('delay', ), ('go_to_end', ), ('go_1.2', ),
        ('iti_all', ), ('iti_first_2', ), ('iti_last_2', )
    ]


@schema
class LinearModelBehaviorModelToFit(dj.Lookup):
    definition = """
    behavior_model:  varchar(30)    # If 'best_aic' etc, the best behavioral model for each session; otherwise --> Model.model_id
    """
    contents = [['best_aic',]]


@schema
class UnitPeriodActivity(dj.Computed):
    definition = """
    -> ephys.Unit
    -> experiment.PeriodForaging
    ---
    trial:          longblob  # Actual trials
    spike_counts:   longblob
    durations:      longblob
    firing_rates:   longblob
    """

    key_source = ephys.Unit & foraging_analysis.SessionTaskProtocol # granularity = unit level
    
    def make(self, key):    
        periods = (experiment.PeriodForaging & 'period NOT IN ("delay_bitcode")').fetch('period')  # 'delay_bitcode' will be used by compute_unit_period_activity if needed

        period_selectivities = []
        for period in periods:
            period_selectivities.append(_compute_unit_period_activity(key, period))
            
        UnitPeriodActivity.insert([{**key, 
                                    **period_sel, 
                                    'period': period} 
                                   for (period_sel, period) in zip(period_selectivities, periods)])
        

    
@schema
class UnitPeriodLinearFit(dj.Computed):
    definition = """
    -> ephys.Unit
    -> LinearModelPeriodToFit
    -> LinearModelBehaviorModelToFit
    -> LinearModel
    ---
    actual_behavior_model:   int
    model_r2=Null:  float   # r square
    model_r2_adj=Null:   float  # r square adj.
    model_p=Null:   float
    model_bic=Null:  float
    model_aic=Null:  float
    """
    
    key_source = (ephys.Unit & foraging_analysis.SessionTaskProtocol - experiment.PhotostimForagingTrial
                 ) * LinearModelBehaviorModelToFit #* LinearModelPeriodToFit # * LinearModel

    class Param(dj.Part):
        definition = """
        -> master
        -> LinearModel.X
        ---
        beta=Null: float
        std_err=Null: float
        p=Null:    float
        t=Null:    float  #  t statistic

        """
        
    def make(self, key):
        # -- Fetech data --
        behavior_model = key['behavior_model']

        # Parse period
        # No longer need this because it has been handled during UnitPeriodActivity
        # if period in ['delay'] and not ephys.TrialEvent & key & 'trial_event_type = "zaberready"':
        #     period = period + '_bitcode'  # Manually correction of bitcodestart to zaberready, if necessary

        # Parse behavioral model_id
        if behavior_model.isnumeric():
            model_id = int(behavior_model)
        else:
            model_id = (foraging_model.FittedSessionModelComparison.BestModel &
                        key & 'model_comparison_idx=0').fetch1(behavior_model)

        # Independent variable is shared across this unit
        all_iv_session = _get_unit_independent_variable(key, model_id=model_id)

        # Add more independent variables, if needed
        all_iv_session['choice_ic_next'] = all_iv_session.choice_ic.shift(-1)
        all_iv_session['trial_normalized'] = all_iv_session.trial / max(all_iv_session.trial)

        trial_session = all_iv_session.trial  # Without ignored trials

        period_activities = (UnitPeriodActivity & key).fetch('period', 'trial', 'firing_rates', as_dict=True)

        to_insert_master = []
        to_insert_part = []

        for key_period in LinearModelPeriodToFit.fetch('KEY'):
            key_1 = {**key, **key_period}
            
            period = key_period['period']
        
            # Get data
            trial_ephys = [x['trial'] for x in period_activities if x['period'] == period][0]
            period_activity = [x['firing_rates'] for x in period_activities if x['period'] == period][0]
                                    
            # TODO Align ephys event with behavior using bitcode! (and save raw bitcodes)
            trial_with_ephys = trial_session <= max(trial_ephys)
            trial = trial_session[trial_with_ephys]  # Truncate behavior trial to max ephys length (this assumes the first trial is aligned, see ingest.ephys)
            all_iv = all_iv_session[trial_with_ephys].copy()  # Also truncate all ivs

            # firing
            firing = pd.DataFrame({f'{period} firing': period_activity[trial - 1]})

            # adding firing history
            for shift in range(1, 11):
                all_iv[f'firing_{shift}_back'] = firing.shift(shift)
                    
            for key_LinearModel in LinearModel.fetch('KEY'):
                key_2 = {**key_1, **key_LinearModel}

                # Parse independent variable
                independent_variables = list((LinearModel.X & key_2).fetch('var_name'))
                if_intercept = (LinearModel & key_2).fetch1('if_intercept')
                
                # -- Fit --
                x = all_iv[independent_variables].astype(float)
                
                nan_indices = x.index[x.isna().any(axis=1)]
                x.drop(nan_indices, inplace=True)
                y = firing.drop(nan_indices)
                
                try:
                    model = sm.OLS(y, sm.add_constant(x) if if_intercept else x)
                    model_fit = model.fit()
                except:
                    print(f'Wrong: {key_2}')
                    return

                # Cache results
                to_insert_master.append({**key_2,
                                        'model_r2': model_fit.rsquared,
                                        'model_r2_adj': model_fit.rsquared_adj,
                                        'model_p': model_fit.f_pvalue,
                                        'model_bic': model_fit.bic if not np.isinf(model_fit.bic) else np.nan,
                                        'model_aic': model_fit.aic if not np.isinf(model_fit.aic) else np.nan,
                                        'actual_behavior_model': model_id})
                
                to_insert_part.extend([{**key_2,
                                        'var_name': para,
                                        'beta': model_fit.params[para],
                                        'std_err': model_fit.bse[para],
                                        'p': model_fit.pvalues[para],
                                        't': model_fit.tvalues[para],
                                        } for para in [p for p in model.exog_names if p!='const']])
            
        # -- Bulk insert --
        self.insert(to_insert_master,
                    skip_duplicates=False)

        self.Param.insert(to_insert_part,
                          skip_duplicates=False)

            
@schema
class UnitTrialAlignedSpikes(dj.Computed):
    definition = """
    -> ephys.Unit
    -> AlignType
    -> experiment.BehaviorTrial
    ---
    aligned_spikes:   longblob   
    """
    
    # Only units that pass unit QC and behavioral QC
    align_type_to_do = AlignType & 'align_type_name IN ("go_cue", "choice", "iti_start")'
    unit_to_do = (ephys.UnitForagingQC & 'unit_minimal_session_qc = 1'
                  & histology.ElectrodeCCFPosition.ElectrodePosition    # With ccf
                  - experiment.PhotostimForagingTrial)  # Without photostim
    key_source = unit_to_do * align_type_to_do
    
#     bin_size = 0.01  # 10 ms window
#     gaussian_sigma = 0.50  # 50 ms half-Gaussian causal filter
    
      
    def make(self, key):
        q_align_type = AlignType & key
        offset, win = q_align_type.fetch1('time_offset', 'psth_win')
        
        # -- Fetch data --
        spike_times = (ephys.Unit & key).fetch1('spike_times')
       
        # Session-wise event times (relative to session start)
        q_events = ephys.TrialEvent & key & {'trial_event_type': q_align_type.fetch1('trial_event_type')}
        trial_keys, events = q_events.fetch('KEY', 'trial_event_time', order_by='trial asc')       
        first_trial_start = ((ephys.TrialEvent & (experiment.Session & key)) 
                             & {'trial_event_type': 'bitcodestart', 'trial': 1}
                            ).fetch1('trial_event_time')
        
        events -= first_trial_start    # Make event times also relative to the first sTrig
        events = events.astype(float)
        events += float(offset)   # Manual correction if necessary (e.g. estimate trialstart from bitcodestart when zaberready is missing)
        
        # -- Align spike times to each event --
        # bins = np.arange(win[0], win[1], UnitAlignedFiring.bin_size)

        # --- Aligned spike count in bins ---
        # spike_count_aligned = np.empty([len(trials), len(bins) - 1], dtype='uint8')
        
        spike_time_aligned = []
        for e_t in events:
            s_t = spike_times[(e_t + win[0] <= spike_times) & (spike_times < e_t + win[1])]
            spike_time_aligned.append(s_t - e_t)
            
        # spike_count_aligned = np.array(list(map(lambda x: np.histogram(x, bins=bins)[0], spike_time_aligned)))
        
        # times = np.mean([bins[:-1], bins[1:]], axis=0)
        
        # --- Insert data (batch for trials) ---
        self.insert([{**key, **trial_key,
                      'aligned_spikes': spike_time_trial}
                     for trial_key, spike_time_trial in zip(trial_keys, spike_time_aligned)],
                     ignore_extra_fields=True)
        
        
@schema
class UnitPSTHChoiceOutcome(dj.Computed):
    definition = """
    -> ephys.Unit 
    -> AlignType
    choice:    varchar(10)   # 'ipsi', 'contra', 'ignore'
    outcome:   varchar(10)   # 'hit', 'miss', 'ignore'
    ---
    raw:    longblob      # spike times in this condition
    trials:     longblob  # trial numbers used in this condition
    psth:      longblob   # binned firing rate (mean +/- sem)
    psth_filtered:   longblob  # filtered by causal half Gaussian (mean +/- sem)
    ts:      longblob  # time centers
    """
    
    key_source = dj.U(*(ephys.Unit.heading.primary_key), 'align_type_name') & UnitTrialAlignedSpikes  # Remove `trial` field

    if_exclude_early_lick=False
    bin_size = 0.01
    gaussian_sigma = 0.05
    
    def make(self, key):
        no_early_lick = '_noearlylick' if UnitPSTHChoiceOutcome.if_exclude_early_lick else ''
        offset, psth_win = (AlignType & key).fetch1('trial_offset', 'psth_win')

        # Get hemi
        try:
            hemi = (ephys.Unit * histology.ElectrodeCCFPosition.ElectrodePosition.proj(hemi='IF(ccf_x > 5739, "left", "right")') & key).fetch1('hemi')
        except:
            hemi = _get_units_hemisphere(key)
            
        ipsi = "L" if hemi == "left" else "R"
        contra = "R" if hemi == "left" else "L"

        # Get trials
        condition_mapping = {('contra', 'hit'): f'{contra}_hit{no_early_lick}',
                             ('contra', 'miss'): f'{contra}_miss{no_early_lick}',
                             ('ipsi', 'hit'): f'{ipsi}_hit{no_early_lick}',
                             ('ipsi', 'miss'): f'{ipsi}_miss{no_early_lick}',
                             ('ignore', 'ignore'): 'ignore',
                            }

        for (choice, outcome), condition_str in condition_mapping.items():
            q_this = TrialCondition.get_trials(condition_str, offset) & key
            trials, spikes_aligned = (UnitTrialAlignedSpikes & key & q_this).fetch('trial', 'aligned_spikes')
            if not len(trials): continue

            bins = np.arange(psth_win[0], psth_win[1], UnitPSTHChoiceOutcome.bin_size)
            ts = np.mean([bins[1:], bins[:-1]], axis=0)
            
            psth = compute_psth_from_spikes_aligned(spikes_aligned, bins, if_filtered=True, 
                                                    sigma=UnitPSTHChoiceOutcome.gaussian_sigma/UnitPSTHChoiceOutcome.bin_size)

           # Batch insert
            self.insert([{**key, 
                         'choice': choice, 
                         'outcome': outcome,
                         'raw': spikes_aligned,
                         'trials': trials,
                         'psth': np.vstack([psth['psth'], psth['sem']]),
                         'psth_filtered': np.vstack([psth['psth_filtered'], psth['sem_filtered']]),
                         'ts': ts}])
        return


import pathlib

report_cfg = dj.config['stores']['report_store']
store_stage = pathlib.Path(report_cfg['stage'])

@schema
class UnitDriftMetric(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    drift_plot:   filepath@report_store
    """
        
    key_source = ephys.Unit & UnitPSTHChoiceOutcome
    align_type = 'iti_start'  # Use firing rates aligned to iti_start
    psth_win = (AlignType & {'align_type_name': align_type}).fetch1('psth_win')
    smooth_win = 6

    class DriftMetric(dj.Part):
        definition = """
        -> master
        method:   varchar(50)  # how to define driftmetric. e.g.: 'poisson_p_choice_outcome', 'poisson_p_all', 'linear_fitting_time_p'
        ---
        drift_metric:   float
        """

    
    def make(self, key):
        """
        Compute driftmetric and generate plots
        """

        units_dir = store_stage / 'unit_drift_metrics'
        units_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 3, figsize=(15, 7), layout='constrained')
        ax = ax.flatten()
        n = 0
           
        poisson_cdf_all = []

        # --- Dave's Poisson p method; grouped by choice/outcome ---
        for choice in ('contra', 'ipsi'):
            for outcome in ('hit', 'miss'):
                this_raw = (UnitPSTHChoiceOutcome & key & {'align_type_name': UnitDriftMetric.align_type, 'choice': choice, 'outcome': outcome}).fetch1('raw')
                this_aver = list(map(lambda x: len(x) / (UnitDriftMetric.psth_win[1] - UnitDriftMetric.psth_win[0]), this_raw))
                poisson_cdf, instability_dave = compute_and_plot_drift_metric(this_aver, 
                                                                         UnitDriftMetric.smooth_win,
                                                                         # int(np.round(len(this_raw)/20)), 
                                                                         ax[n])
                ax[n].set_title(f'{choice} {outcome}, instab = {instability_dave:.2f}')
                n += 1
                
                poisson_cdf_all.append(poisson_cdf)
            
        poisson_cdf_all = np.hstack(poisson_cdf_all)
        instability_all = np.logical_or(poisson_cdf_all > 0.95, poisson_cdf_all < 0.05).sum() / len(poisson_cdf_all)
        # print(f"Dave's drift metric, grouped by choice x outcome: {instability_all}")
            
        # --- Not grouped (native Dave method) ---
        this_raw = (UnitTrialAlignedSpikes & key & 'align_type_name = "iti_start"').fetch('aligned_spikes', order_by='trial')
        this_aver = list(map(lambda x: len(x) / (UnitDriftMetric.psth_win[1] - UnitDriftMetric.psth_win[0]), this_raw))
        _, instability_dave = compute_and_plot_drift_metric(this_aver,
                                                                 UnitDriftMetric.smooth_win,
                                                                 # int(np.round(len(this_raw)/20)), 
                                                                 ax[-1])
        
        ax[-1].set_title(f'grouped = {instability_all:.2f}, not grouped: {instability_dave:.2f}')
        fig.suptitle(key)
        # print(f"not grouped: {instability}")
                
        fn_prefix = f'{instability_all:0.4f}_{key["subject_id"]}_{key["session"]}_{key["insertion_number"]}_{key["unit"]:03}_'
        fn_prefix = "".join( x for x in fn_prefix if (x.isalnum() or x in "._- "))  # turn to valid file name
        
        fig_dict = save_figs(
            (fig,),
            ('drift_plot',),
            units_dir, fn_prefix)
 
        plt.close('all')
        self.insert1({**key, **fig_dict})
        self.DriftMetric.insert([{**key, 'method': 'poisson_p_choice_outcome', 'drift_metric': instability_all}, 
                                 {**key, 'method': 'poisson_p_dave', 'drift_metric': instability_dave}])
    
        
        return
        

# ============= Helpers =============
def compute_psth_from_spikes_aligned(spikes_aligned, bins, if_filtered=True, sigma=None):   
    # PSTH
    psth_per_trial = np.vstack([np.histogram(trial_spike, bins=bins)[0] / UnitPSTHChoiceOutcome.bin_size for trial_spike in spikes_aligned])    
    psth = np.mean(psth_per_trial, axis=0)
    sem = np.std(psth_per_trial, axis=0) / np.sqrt(len(spikes_aligned))
        
    if not if_filtered:
        return dict(psth=psth, sem=sem)

    # Gaussian filter
    psth_per_trial_filtered = halfgaussian_filter1d(psth_per_trial, 
                                                    sigma=sigma)
    psth_filtered = np.mean(psth_per_trial_filtered, axis=0)
    sem_filtered = np.std(psth_per_trial_filtered, axis=0) / np.sqrt(len(spikes_aligned))
    
    return dict(psth=psth, sem=sem, psth_filtered=psth_filtered, sem_filtered=sem_filtered)



def compute_unit_psth_and_raster(unit_key, trial_keys, align_type='go_cue', bin_size=0.04):
    """
    Align spikes of specified unit and trial-set to specified align_event_type,
    compute psth with specified window and binsize, and generate data for raster plot.
    (for foraging task only)

    @param unit_key: key of a single unit to compute the PSTH for
    @param trial_keys: list of all the trial keys to compute the PSTH over
    @param align_type: psth_foraging.AlignType
    @param bin_size: (in sec)
    
    Returns a dictionary of the form:
      {
         'bins': time bins,
         'trials': ephys.Unit.TrialSpikes.trials,
         'spikes_aligned': aligned spike times per trial
         'psth': (bins x 1)
         'psth_per_trial': (trial x bins)
         'raster': Spike * Trial raster [np.array, np.array]
      }
    """
    
    # import time; s = time.time()
    
    q_align_type = AlignType & {'align_type_name': align_type}
    
    # -- Get global times for spike and event --
    q_spike = ephys.Unit & unit_key  # Using ephys.Unit, not ephys.Unit.TrialSpikes
    q_event = ephys.TrialEvent & trial_keys & {'trial_event_type': q_align_type.fetch1('trial_event_type')}   # Using ephys.TrialEvent, not experiment.TrialEvent
    
    if not q_spike or not q_event:
        return None

    # Session-wise spike times (relative to the first sTrig, i.e. 'bitcodestart'. see line 212 of ingest.ephys)
    # ss = time.time()
    spikes = q_spike.fetch1('spike_times')
    # print(f'fetch_spikes: {time.time()-ss}')
    
    # Session-wise event times (relative to session start)
    events, trials = q_event.fetch('trial_event_time', 'trial', order_by='trial asc')
    # Make event times also relative to the first sTrig
    events -= (ephys.TrialEvent & trial_keys.proj(_='trial') & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time')
    events = events.astype(float)
    
    # Manual correction of trialstart, if necessary
    events += q_align_type.fetch('time_offset').astype(float)
    
    # -- Align spike times to each event --
    win = q_align_type.fetch1('psth_win')
    spikes_aligned = []
    for e_t in events:
        s_t = spikes[(e_t + win[0] <= spikes) & (spikes < e_t + win[1])]
        spikes_aligned.append(s_t - e_t)
    
    # -- Compute psth --
    binning = np.arange(win[0], win[1], bin_size)
    
    # psth (bins x 1)
    all_spikes = np.concatenate(spikes_aligned)
    psth, edges = np.histogram(all_spikes, bins=binning)
    psth = psth / len(q_event) / bin_size
    
    # psth per trial (trial x bins)
    psth_per_trial = np.vstack([np.histogram(trial_spike, bins=binning)[0] / bin_size for trial_spike in spikes_aligned])

    # raster (all spike time, all trial number)
    raster = [all_spikes,
              np.concatenate([[t] * len(s)
                              for s, t in zip(spikes_aligned, trials)])]

    # print(f'compute_unit_psth_and_raster: {time.time() - s}')

    return dict(bins=binning[1:], trials=trials, spikes_aligned=spikes_aligned,
                psth=psth, psth_per_trial=psth_per_trial, raster=raster)


def _compute_unit_period_activity(unit_key, period):
    """
    Given unit and period, compute average firing rate over trials
    I tried to put this in a table, but it's too slow... (too many elements)
    @param unit_key:
    @param period: -> experiment.PeriodForaging, or arbitrary list in the same format
    @return: DataFrame(trial, spike_count, duration, firing_rate)
    """

    q_spike = ephys.Unit & unit_key
    q_event = ephys.TrialEvent & unit_key
    if not q_spike or not q_event:
        return None

    # for (the very few) sessions without zaber feedback signal, use 'bitcodestart' with manual correction
    if period == 'delay' and \
            not q_event & 'trial_event_type = "zaberready"':
        period = 'delay_bitcode'

    # -- Fetch global session times of given period, for each trial --
    try:
        (start_event_type, start_trial_shift, start_time_shift,
         end_event_type, end_trial_shift, end_time_shift) = (experiment.PeriodForaging & {'period': period}
                  ).fetch1('start_event_type', 'start_trial_shift', 'start_time_shift',
                           'end_event_type', 'end_trial_shift', 'end_time_shift')
    except:
        (start_event_type, start_trial_shift, start_time_shift,
         end_event_type, end_trial_shift, end_time_shift) = period
    
    start = {k['trial']: float(k['start_event_time'])
             for k in (q_event & {'trial_event_type': start_event_type}).proj(
            start_event_time=f'trial_event_time + {start_time_shift}').fetch(as_dict=True)}
    end = {k['trial']: float(k['end_event_time'])
             for k in (q_event & {'trial_event_type': end_event_type}).proj(
            end_event_time=f'trial_event_time + {end_time_shift}').fetch(as_dict=True)}

    # Handle edge effects due to trial shift
    trials = np.array(list(start.keys()))
    actual_trials = trials[(trials <= max(trials) - end_trial_shift) &
                           (trials >= min(trials) - start_trial_shift)]

    # -- Fetch and count spikes --
    spikes = q_spike.fetch1('spike_times')
    
    # !!!Very important: ephys.TrialEvent is relative to the session start, whereas spike times are relative to the first sTrig...!!!
    first_bit_code_start = float((ephys.TrialEvent & unit_key & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time'))
    spikes = spikes + first_bit_code_start

    spike_counts, durations = [], []

    for trial in actual_trials:
        t_s = start[trial + start_trial_shift]
        t_e = end[trial + end_trial_shift]

        spike_counts.append(((t_s <= spikes) & (spikes < t_e)).sum())  # Much faster than sum(... & ...) (python sum on np array)!!
        durations.append(t_e - t_s)

    return {'trial': actual_trials, 'spike_counts': np.array(spike_counts),
            'durations': np.array(durations), 'firing_rates': np.array(spike_counts) / np.array(durations)}


import scipy.ndimage

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


from scipy.stats import poisson

def compute_poisson_p(aver_firing_per_trial, window_size=6, ds_factor=6):
    """
    Dave Liu's method
    Major problem: the result highly depends on the window size
    """
    mean_spike_rate = np.mean(aver_firing_per_trial)
    # -- moving-average
    kernel = np.ones(window_size) / window_size
    processed_trial_spike_rates = np.convolve(aver_firing_per_trial, kernel, 'valid')
    # -- down-sample
    processed_trial_spike_rates = processed_trial_spike_rates[::ds_factor]
    # -- compute drift_qc from poisson distribution
    poisson_cdf = poisson.cdf(processed_trial_spike_rates, mean_spike_rate)
    instability = np.logical_or(poisson_cdf > 0.95, poisson_cdf < 0.05).sum() / len(poisson_cdf)
    return np.array(poisson_cdf), instability, processed_trial_spike_rates


def compute_and_plot_drift_metric(this_aver, win_size, ax):
    ax.plot(np.linspace(0, 1, len(this_aver)), np.array(this_aver), 'b', lw=0.5)
    
    poisson_cdf, instability, smoothed_firing = compute_poisson_p(this_aver, window_size=win_size, 
                                                                  ds_factor=win_size)
    # p = kstest(this_aver, 'norm')
    # print(p.pvalue)
    xx = np.linspace(0, 1, len(poisson_cdf))    
    ax.plot(xx, smoothed_firing, 'b-')
    ax.axhline(np.mean(this_aver), c='b', ls='--')

    idx_sig = np.logical_or(poisson_cdf > 0.95, poisson_cdf < 0.05)
    ax2 = ax.twinx()
    ax2.plot(xx, poisson_cdf, 'ko-')
    ax2.plot(xx[idx_sig], poisson_cdf[idx_sig], 'ro')
    ax2.set_ylim([0, 1])
    
    return poisson_cdf, instability

def save_figs(figs, fig_names, dir2save, prefix):
    fig_dict = {}
    for fig, figname in zip(figs, fig_names):
        fig_fp = dir2save / (prefix + figname + '.png')
        fig.tight_layout()
        fig.savefig(fig_fp, facecolor=fig.get_facecolor())
        print(f'Generated {fig_fp}')
        fig_dict[figname] = fig_fp.as_posix()

    return fig_dict