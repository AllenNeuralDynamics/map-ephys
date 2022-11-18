import datajoint as dj
import pathlib
import numpy as np
import json
from datetime import datetime
from dateutil.tz import tzlocal
from datajoint.errors import DataJointError
import pynwb
from pynwb import NWBFile, NWBHDF5IO

from pipeline import lab, experiment, tracking, ephys, histology, psth, ccf, foraging_model
from pipeline.util import _get_session_independent_variable
from pipeline.report import get_wr_sessdatetime
from pipeline.ingest import ephys as ephys_ingest
from pipeline.ingest import tracking as tracking_ingest

# ephys_root_data_dir = pathlib.Path(get_ephys_paths()[0])
# tracking_root_data_dir = pathlib.Path(tracking_ingest.get_tracking_paths()[0])


# Helper functions for raw ephys data import
def get_electrodes_mapping(electrodes):
    """
    Create a mapping from the probe and electrode id to the row number of the electrodes
    table. This is used in the construction of the DynamicTableRegion that indicates what rows of the electrodes
    table correspond to the data in an ElectricalSeries.
    Parameters
    ----------
    electrodes: hdmf.common.table.DynamicTable
    Returns
    -------
    dict
    """
    return {
        (electrodes["group"][idx].device.name, electrodes["id"][idx],): idx
        for idx in range(len(electrodes))
    }


def gains_helper(gains):
    """
    This handles three different cases for gains:
    1. gains are all 1. In this case, return conversion=1e-6, which applies to all
    channels and converts from microvolts to volts.
    2. Gains are all equal, but not 1. In this case, multiply this by 1e-6 to apply this
    gain to all channels and convert units to volts.
    3. Gains are different for different channels. In this case use the
    `channel_conversion` field in addition to the `conversion` field so that each
    channel can be converted to volts using its own individual gain.
    Parameters
    ----------
    gains: np.ndarray
    Returns
    -------
    dict
        conversion : float
        channel_conversion : np.ndarray
    """
    if all(x == 1 for x in gains):
        return dict(conversion=1e-6, channel_conversion=None)
    if all(x == gains[0] for x in gains):
        return dict(conversion=1e-6 * gains[0], channel_conversion=None)
    return dict(conversion=1e-6, channel_conversion=gains)
    

# Some constants to work with
zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available


def datajoint_to_nwb(session_key, raw_ephys=False, raw_video=False):
    """
    Generate one NWBFile object representing all data
    coming from the specified "session_key" (representing one session)
    """
    water_res_num, sess_datetime = get_wr_sessdatetime(session_key)

    session_identifier = f'{water_res_num}_{sess_datetime}_s{session_key["session"]}'

    experiment_description = (experiment.TaskProtocol
                            & (experiment.BehaviorTrial & session_key)).fetch1(
        'task_protocol_description')

    try:
        session_descr = (experiment.SessionComment & session_key).fetch1('session_comment')
    except DataJointError:
        session_descr = ''

    nwbfile = NWBFile(identifier=session_identifier,
                    session_description=session_descr,
                    session_start_time=datetime.strptime(sess_datetime, '%Y%m%d_%H%M%S'),
                    file_create_date=datetime.now(tzlocal()),
                    experimenter=list((experiment.Session & session_key).fetch('username')),
                    data_collection='',
                    institution='Janelia Research Campus / Allen Institute',
                    experiment_description=experiment_description,
                    related_publications='',
                    keywords=[])

    # ==================================== SUBJECT ==================================
    subject = (lab.Subject * lab.WaterRestriction.proj('water_restriction_number') & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subject['subject_id']),
        date_of_birth=datetime.combine(subject['date_of_birth'], zero_time) if subject['date_of_birth'] else None,
        description=subject['water_restriction_number'],
        sex=subject['sex'],
        species='mus musculus')

    # ==================================== EPHYS ==================================
    # add additional columns to the electrodes table
    electrodes_query = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
    for additional_attribute in ['shank', 'shank_col', 'shank_row']:
        nwbfile.add_electrode_column(
            name=electrodes_query.heading.attributes[additional_attribute].name,
            description=electrodes_query.heading.attributes[additional_attribute].comment)

    nwbfile.add_electrode_column(name='electrode_id', description='electrode id on the probe')  # Don't use id to avoid confusion
    nwbfile.add_electrode_column(name='ccf_annotation', description='ccf annotation')  # Don't use id to avoid confusion

    # unit_qc
    unit_qc = {'minimal': 'unit_amp > 70 '
                   'AND avg_firing_rate > 0.1 '
                   'AND presence_ratio > 0.9 '
                   'AND isi_violation < 0.1 '
                   'AND amplitude_cutoff < 0.15'}
    qc_in_use = unit_qc['minimal']
    
    nwbfile.units = pynwb.file.Units(name='units', description=f'unit qc = {qc_in_use}')


    # add additional columns to the units table
    if dj.__version__ >= '0.13.0':
        units_query = (ephys.ProbeInsertion.RecordingSystemSetup
                    * ephys.Unit & session_key).proj(..., '-spike_sites', '-spike_depths').join(    # Remove spike_sites and depths to avoid inhomogeneous error
            ephys.UnitStat, left=True).join(
            ephys.MAPClusterMetric.DriftMetric, left=True).join(
            ephys.ClusterMetric, left=True).join(
            ephys.WaveformMetric, left=True)
    else:
        units_query = (ephys.ProbeInsertion.RecordingSystemSetup
                    * ephys.Unit & session_key).proj(...).aggr(
            ephys.UnitStat, ..., **{n: n for n in ephys.UnitStat.heading.names if n not in ephys.UnitStat.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.MAPClusterMetric.DriftMetric, ..., **{n: n for n in ephys.MAPClusterMetric.DriftMetric.heading.names if n not in ephys.MAPClusterMetric.DriftMetric.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.ClusterMetric, ..., **{n: n for n in ephys.ClusterMetric.heading.names if n not in ephys.ClusterMetric.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.WaveformMetric, ..., **{n: n for n in ephys.WaveformMetric.heading.names if n not in ephys.WaveformMetric.heading.primary_key},
            keep_all_rows=True)

    units_query = units_query & qc_in_use

    units_omitted_attributes = ['subject_id', 'session',
                                'clustering_method', 'unit', 'unit_uid', 'probe_type',
                                'epoch_name_quality_metrics', 'epoch_name_waveform_metrics',
                                'electrode_config_name', 'electrode_group',
                                'electrode', 'waveform', 'annotation_version']

    for attr in units_query.heading.names:
        if attr in units_omitted_attributes + ['spike_times']:   # Don't add column `spike_times` to avoid inhomogeneous error (leave it as nwb internal data type that enables DynamicTable (spike_times + spike_times_index))
            continue
        nwbfile.add_unit_column(
            name=units_query.heading.attributes[attr].name,
            description=units_query.heading.attributes[attr].comment)
        
    nwbfile.add_unit_column(name='unit_id', description='original unit_id on one probe')  # Don't use id to avoid confusion

    # iterate through curated clusterings and export units data
    for insert_key in (ephys.ProbeInsertion & session_key).fetch('KEY'):
    # ---- Probe Insertion Location ----
        if ephys.ProbeInsertion.InsertionLocation & insert_key:
            insert_location = {
                k: str(v) for k, v in (ephys.ProbeInsertion.InsertionLocation.proj()
                                        & insert_key).aggr(
                    ephys.ProbeInsertion.RecordableBrainRegion.proj(
                        ..., brain_region='CONCAT(hemisphere, " ", brain_area)'),
                    ..., brain_regions='GROUP_CONCAT(brain_region SEPARATOR ", ")').fetch1().items()
                if k not in ephys.ProbeInsertion.primary_key}
            insert_location = json.dumps(insert_location)
        else:
            insert_location = 'N/A'

        # ---- Electrode Configuration ----
        electrode_config = (lab.Probe * lab.ProbeType * lab.ElectrodeConfig
                            * ephys.ProbeInsertion & insert_key).fetch1()
        ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
        ephys_device = (nwbfile.get_device(ephys_device_name)
                        if ephys_device_name in nwbfile.devices
                        else nwbfile.create_device(name=ephys_device_name))

        electrode_group = nwbfile.create_electrode_group(
            name=f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}',
            description=json.dumps(electrode_config, default=str),
            device=ephys_device,
            location=insert_location)

        electrode_query = (lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
                            & electrode_config)
        electrode_ccf = {e: {'x': float(z), 'y': float(y), 'z': 5739 * 2 - float(x),    # Turn DataJoint ccf to real allen_ccf
                             'ccf_annotation': annot} for e, x, y, z, annot in zip(
            * (histology.ElectrodeCCFPosition.ElectrodePosition * ccf.CCFAnnotation 
               & electrode_config).fetch(
                'electrode', 'ccf_x', 'ccf_y', 'ccf_z', 'annotation'))}

        for electrode in electrode_query.fetch(as_dict=True):
            nwbfile.add_electrode(
                electrode_id=electrode['electrode'], group=electrode_group,
                filtering='', imp=-1.,
                **electrode_ccf.get(electrode['electrode'], {'x': np.nan, 'y': np.nan, 'z': np.nan, 'ccf_annotation': 'unknown'}),
                rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
                shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
                location=electrode_group.location)

        electrode_df = nwbfile.electrodes.to_dataframe()
        electrode_ind = electrode_df.electrode_id[electrode_df.group_name == electrode_group.name]  # index: index in nwbfile.electrodes; value: original electrode index in that probe

        # ---- Units ----
        unit_query = units_query & insert_key
        for unit in unit_query.fetch(as_dict=True, order_by='unit'):
            # make an electrode table region (which electrode(s) is this unit coming from)
            unit['unit_id'] = unit.pop('unit')
            unit['electrodes'] = electrode_ind.index[electrode_ind == unit.pop('electrode')].values

            unit['electrode_group'] = electrode_group
            unit['waveform_mean'] = unit.pop('waveform')
            unit['waveform_sd'] = np.full(1, np.nan)

            for attr in list(unit.keys()):
                if attr in units_omitted_attributes:
                    unit.pop(attr)
                elif unit[attr] is None:
                    unit[attr] = np.nan

            nwbfile.add_unit(**unit)


        # ---- Raw Ephys Data ---
        if raw_ephys:
            ks_dir_relpath = (ephys_ingest.EphysIngest.EphysFile.proj(
                ..., insertion_number='probe_insertion_number')
                              & insert_key).fetch('ephys_file')
            ks_dir = ephys_root_data_dir / ks_dir_relpath
            npx_dir = ks_dir.parent

            try:
                next(npx_dir.glob('*imec*.ap.bin'))
            except StopIteration:
                raise FileNotFoundError(f'No raw ephys file (.ap.bin) found at {npx_dir}')

            sampling_rate = (ephys.ProbeInsertion.RecordingSystemSetup & insert_key).fetch1('sampling_rate')
            probe_id, probe_type = (ephys.ProbeInsertion & insert_key).fetch1('probe', 'probe_type')
            mapping = get_electrodes_mapping(nwbfile.electrodes)

            extractor = extractors.read_spikeglx(npx_dir)

            conversion_kwargs = gains_helper(extractor.get_channel_gains())

            recording_channels_by_id = (lab.ElectrodeConfig.Electrode * ephys.ProbeInsertion
                                        & insert_key).fetch('electrode')

            probe_str = f'{probe_id} ({probe_type})'

            nwbfile.add_acquisition(
                pynwb.ecephys.ElectricalSeries(
                    name=f"ElectricalSeries{insert_key['insertion_number']}",
                    description=f"Ephys recording from {probe_str}, at location: {insert_location}",
                    data=SpikeInterfaceRecordingDataChunkIterator(extractor),
                    rate=float(sampling_rate),
                    electrodes=nwbfile.create_electrode_table_region(
                        region=[mapping[(probe_str, x)] for x in recording_channels_by_id],
                        name="electrodes",
                        description="recorded electrodes",
                    ),
                    **conversion_kwargs
                )
            )

    # =============================== PHOTO-STIMULATION ===============================
    stim_sites = {}
    photostim_query = (experiment.Photostim & (experiment.PhotostimTrial & session_key))
    if photostim_query:
        for photostim_key in photostim_query.fetch('KEY'):
            photostim = (experiment.Photostim * lab.PhotostimDevice.proj('excitation_wavelength') & photostim_key).fetch1()
            stim_device = (nwbfile.get_device(photostim['photostim_device'])
                        if photostim['photostim_device'] in nwbfile.devices
                        else nwbfile.create_device(name=photostim['photostim_device']))

            stim_site = pynwb.ogen.OptogeneticStimulusSite(
                name=f'{photostim["photostim_device"]}_{photostim["photo_stim"]}',
                device=stim_device,
                excitation_lambda=float(photostim['excitation_wavelength']),
                location=json.dumps([{k: v for k, v in stim_locs.items()
                                    if k not in experiment.Photostim.primary_key}
                                    for stim_locs in (experiment.Photostim.PhotostimLocation
                                                    & photostim_key).fetch(as_dict=True)], default=str),
                description=f'excitation_duration: {photostim["duration"]}')
            nwbfile.add_ogen_site(stim_site)
            stim_sites[photostim['photo_stim']] = stim_site 

    # =============================== TRACKING =============================== 
    # if tracking.Tracking & session_key:
    #     behav_acq = pynwb.behavior.BehavioralTimeSeries(name='BehavioralTimeSeries')
    #     nwbfile.add_acquisition(behav_acq)

    #     tracking_devices = (tracking.TrackingDevice & (tracking.Tracking & session_key)).fetch(as_dict=True)

    #     for trk_device in tracking_devices:
    #         trk_device_name = trk_device['tracking_device'].replace(' ', '') + '_' + trk_device['tracking_position']
    #         trk_fs = float(trk_device['sampling_rate'])
    #         for feature, feature_tbl in tracking.Tracking().tracking_features.items():
    #             ft_attrs = [n for n in feature_tbl.heading.names if n not in feature_tbl.primary_key]
    #             if feature_tbl & trk_device & session_key:
    #                 if feature == 'WhiskerTracking':
    #                     additional_conditions = [{'whisker_name': n} for n in
    #                                              set((feature_tbl & trk_device & session_key).fetch(
    #                                                  'whisker_name'))]
    #                 else:
    #                     additional_conditions = [{}]
    #                 for r in additional_conditions:
    #                     samples, start_time, *position_data = (experiment.SessionTrial
    #                                                            * tracking.Tracking
    #                                                            * feature_tbl
    #                                                            & session_key
    #                                                            & r).fetch(
    #                         'tracking_samples', 'start_time', *ft_attrs, order_by='trial')

    #                     tracking_timestamps = np.hstack([np.arange(nsample) / trk_fs + float(trial_start_time)
    #                                                      for nsample, trial_start_time in zip(samples, start_time)])
    #                     position_data = np.vstack([np.hstack(d) for d in position_data])

    #                     behav_ts_name = f'{trk_device_name}_{feature}' + (f'_{r["whisker_name"]}' if r else '')

    #                     behav_acq.create_timeseries(name=behav_ts_name,
    #                                                 data=position_data,
    #                                                 timestamps=tracking_timestamps,
    #                                                 description=f'Time series for {feature} position: {tuple(ft_attrs)}',
    #                                                 unit='a.u.',
    #                                                 conversion=1.0)

    # =============================== BEHAVIOR TRIALS ===============================     
    q_photostim = (experiment.PhotostimEvent
                   * experiment.Photostim & session_key).proj(
        'photostim_event_time', 'power', 'duration')
    q_trial = experiment.SessionTrial * experiment.BehaviorTrial & session_key 
    q_trial *= experiment.WaterPortChoice.proj(choice='water_port')  # Add choice
    
    # reward schedule
    q_p_reward = experiment.BehaviorTrial.proj() * experiment.SessionBlock.BlockTrial * experiment.SessionBlock.WaterPortRewardProbability & session_key
    p_reward = np.vstack([(q_p_reward & 'water_port="left"').fetch('reward_probability', order_by='trial').astype(float),
                          (q_p_reward & 'water_port="right"').fetch('reward_probability', order_by='trial').astype(float)])
 
    
    model_id = 20  # Hard-coded model_id (Hattori with choice kernel)
    
    _, df_latent = _get_session_independent_variable(session_key, model_id=model_id)
    
    # q_trial = q_trial.aggr(
    #     q_photostim, ...,
    #     photostim_onset='IFNULL(GROUP_CONCAT(photostim_event_time SEPARATOR ", "), "N/A")',
    #     photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
    #     photostim_duration='IFNULL(GROUP_CONCAT(duration SEPARATOR ", "), "N/A")',
    #     keep_all_rows=True)

    skip_adding_columns = experiment.Session.primary_key 

    if q_trial:
        # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {tag: {'name': tag,
                               'description': q_trial.heading.attributes[tag].comment}
                         for tag in q_trial.heading.names
                         if tag not in skip_adding_columns + ['start_time', 'stop_time']}
        
        # Add new table columns to nwb trial-table
        for column in trial_columns.values():
            nwbfile.add_trial_column(**column)
      
        nwbfile.add_trial_column(name='left_reward_prob', description=f'reward refill prob. of the left port')
        nwbfile.add_trial_column(name='right_reward_prob', description=f'reward refill prob. of the right port')
        nwbfile.add_trial_column(name='left_action_value', description=f'left action value after each trial, from model {model_id}')
        nwbfile.add_trial_column(name='right_action_value', description=f'right action value after each trial, from model {model_id}')
        nwbfile.add_trial_column(name='rpe', description=f'reward prediction error of each trial, from model {model_id}')

        # Add entries to the trial-table
        for trial in q_trial.fetch(as_dict=True):
            trial['start_time'], trial['stop_time'] = float(trial['start_time']), float(trial['stop_time'])
            if trial['choice'] is None:
                trial['choice'] = 'null'
            
            trial['left_reward_prob'], trial['right_reward_prob'] = p_reward[:, trial['trial'] - 1]
            
            if any(df_latent.trial==trial['trial']):
                trial['left_action_value'], trial['right_action_value'], trial['rpe'] = df_latent.loc[df_latent.trial==trial['trial'], ['left_action_value', 'right_action_value', 'rpe']].values[0]
            else:
                trial['left_action_value'], trial['right_action_value'], trial['rpe'] = [np.nan] * 3  # Ignored trial
            
            nwbfile.add_trial(**{k: v for k, v in trial.items() if k not in skip_adding_columns})


    # =============================== BEHAVIOR TRIALS' EVENTS ===============================

    behavioral_event = pynwb.behavior.BehavioralEvents(name='BehavioralEvents')
    nwbfile.add_acquisition(behavioral_event)

    # ---- behavior events
    q_trial_event = ephys.TrialEvent & session_key
    first_trial_bitcode_start = (q_trial_event & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time')
    q_trial_event = q_trial_event.proj(trial_event_type='trial_event_type', trial_event_time=f'trial_event_time - {first_trial_bitcode_start}')  # Align with ephys: 0 = first trial start
    unique_trial_event_type = (experiment.TrialEventType & q_trial_event).fetch('trial_event_type')
    
    trial_event_to_exclude = ['bitcodestart', 'bpodstart', 'zaberstep']

    for trial_event_type in unique_trial_event_type:
        if trial_event_type in trial_event_to_exclude:
            continue
        
        trial, trial_event_time = (q_trial_event & {'trial_event_type': trial_event_type}).fetch(
            'trial', 'trial_event_time', order_by='trial')
        
        behavioral_event.create_timeseries(
            name='lickportready' if trial_event_type == 'zaberready' else trial_event_type,
            unit='a.u.', conversion=1.0,
            data=np.full_like(trial_event_time.astype(float), 1),
            timestamps=trial_event_time.astype(float),
            description='time (second) relative to the first trial start (aligned with ephys)')         

    # If there is no zaber feedback, estimate `lickportready` using bitcodestart
    # see https://github.com/hanhou/map-ephys/blob/26892a06b5877687ba4c95d23cbefa5ab4266033/pipeline/psth_foraging.py#L379
    offset_bitcodestart_to_zaberready = 0.146
    
    if 'zaberready' not in unique_trial_event_type:
        bitcodestart_time = (q_trial_event & {'trial_event_type': 'bitcodestart'}).fetch(
                                              'trial_event_time', order_by='trial')
        
        behavioral_event.create_timeseries(
            name='lickportready',
            unit='a.u.', conversion=1.0,
            data=np.full_like(bitcodestart_time.astype(float), 1),
            timestamps=bitcodestart_time.astype(float) + offset_bitcodestart_to_zaberready,
            description='time (second) relative to the first trial start (aligned with ephys)') 
        

    # ---- action events

    q_action_event = ephys.ActionEvent & session_key
    q_action_event = q_action_event.proj(action_event_type='action_event_type', action_event_time=f'action_event_time - {first_trial_bitcode_start}')  # Align with ephys: 0 = first trial start
    unique_action_event_type = (experiment.ActionEventType & q_action_event).fetch('action_event_type')

    for action_event_type in unique_action_event_type:
        trial, action_event_time = (q_action_event
                               & {'action_event_type': action_event_type}).fetch(
            'trial', 'action_event_time', order_by='trial')

        behavioral_event.create_timeseries(
            name=action_event_type.replace(' ', '_'),
            unit='a.u.', conversion=1.0,
            data=np.full_like(action_event_time.astype(float), 1),
            timestamps=action_event_time.astype(float),
            description='time (second) relative to the first trial start (aligned with ephys)')
        
    # If no lick time in ephys NI, use bpod time
    if 'left lick' not in unique_action_event_type:
        first_trial_bitcode_start = (ephys.TrialEvent & session_key & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time')
        q_action_global_from_bpod = ((experiment.ActionEvent & session_key)
                                     * (experiment.TrialEvent & 'trial_event_type = "go"').proj(go_bpod='trial_event_time')
                                     * (ephys.TrialEvent & 'trial_event_type = "go"').proj(go_global='trial_event_time')
                                     ).proj(..., action_event_time_global=f'action_event_time - go_bpod + go_global - {first_trial_bitcode_start}')

        for action_event_type in ['left lick', 'right lick']:
            action_event_time_global_from_bpod = (q_action_global_from_bpod & {'action_event_type': action_event_type}).fetch('action_event_time_global')
            
            behavioral_event.create_timeseries(
                name=action_event_type.replace(' ', '_'),
                unit='a.u.', conversion=1.0,
                data=np.full_like(action_event_time_global_from_bpod.astype(float), 1),
                timestamps=action_event_time_global_from_bpod.astype(float),
                description='time (second) relative to the first trial start (aligned with ephys)')
    
    

    # # ---- photostim events ----

    # q_photostim_event = (experiment.PhotostimEvent
    #                      * experiment.Photostim.proj('duration')
    #                      * experiment.SessionTrial
    #                      & session_key).proj(
    #     'trial', 'power', 'photostim_event_time',
    #     event_start='photostim_event_time + start_time',
    #     event_stop='photostim_event_time + start_time + duration')

    # trials, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
    #     'trial', 'event_start', 'event_stop', 'power', 'photo_stim', order_by='trial')

    # behavioral_event.create_timeseries(
    #     name='photostim_start_times', unit='mW', conversion=1.0,
    #     description='Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied',
    #     data=powers.astype(float),
    #     timestamps=event_starts.astype(float),
    #     control=photo_stim.astype('uint8'), control_description=stim_sites)
    # behavioral_event.create_timeseries(
    #     name='photostim_stop_times', unit='mW', conversion=1.0,
    #     description='Timestamps of the photo-stimulation being switched off',
    #     data=np.full_like(event_starts.astype(float), 0),
    #     timestamps=event_stops.astype(float),
    #     control=photo_stim.astype('uint8'), control_description=stim_sites)

    # ----- Raw Video Files -----
    if raw_video:
        tracking_files_info = (tracking_ingest.TrackingIngest.TrackingFile & session_key).fetch(
            as_dict=True, order_by='tracking_device, trial')
        for tracking_file_info in tracking_files_info:
            video_path = tracking_root_data_dir / tracking_file_info.pop('tracking_file')
            video_metadata = dict(
                Behavior=dict(
                    Movies=[
                        dict(
                            name=video_path,
                            description=video_path.as_posix(),
                            unit="n.a.",
                            format='external',
                            starting_frame=[0, 0, 0],
                            comments=str(tracking_file_info))
                    ]
                )
            )
            MovieInterface([video_path]).run_conversion(nwbfile=nwbfile, metadata=video_metadata, external_mode=False)
    
    return nwbfile


def export_recording(session_keys, output_dir='./', overwrite=False):
    if not isinstance(session_keys, list):
        session_keys = [session_keys]

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for session_key in session_keys:
        
        # Write to .nwb
        water_res_num, sess_datetime = get_wr_sessdatetime(session_key)
        session_identifier = f'{water_res_num}_{sess_datetime}_s{session_key["session"]}'
        save_file_name = ''.join([session_identifier, '.nwb'])
        output_fp = (output_dir / save_file_name).absolute()
        
        if overwrite or not output_fp.exists():
            nwbfile = datajoint_to_nwb(session_key)
            with NWBHDF5IO(output_fp.as_posix(), mode='w') as io:
                io.write(nwbfile)
                print(f'\tWrite NWB 2.0 file: {save_file_name}')
        else:
            print(f'skip {save_file_name}')
                

##### WORKAROUND FOR NWB IO BUG #############               
                
import pandas as pd
import h5py
                
                
def import_nwb_to_df(filename):
    with h5py.File(filename, mode='r') as f:
        g_trials = f['intervals']['trials']
        g_events = f['acquisition']['BehavioralEvents']
        g_electrodes = f['general']['extracellular_ephys']['electrodes']
        g_units = f['units']        
        
        df_trials = hdf_group_to_df(g_trials)
        dict_events = {key: g_events[key]['timestamps'][:] for key in g_events}

        df_units = hdf_group_to_df(g_units)
        df_electrodes = hdf_group_to_df(g_electrodes)
       
    # Add electrode info to units table
    df_units = df_units.merge(df_electrodes, left_on='electrodes', right_on='id')
    
    # Add hemisphere
    df_units['hemisphere'] = np.where(df_units['z'] <= 5739, 'left', 'right')
    
    return [df_trials, dict_events, df_units]

        
def hdf_group_to_df(group):
    '''
    Convert h5py group to dataframe
    '''
    
    tmp_dict = {}
    
    # Handle spike_times
    # see https://nwb-schema.readthedocs.io/en/latest/format_description.html#tables-and-ragged-arrays
    if 'spike_times' in group.keys():
        spike_times, spike_times_index = group['spike_times'], group['spike_times_index']
        unit_spike_times = []
        for i, idx in enumerate(spike_times_index):
            start_idx = 0 if i == 0 else spike_times_index[i - 1]
            unit_spike_times.append(spike_times[start_idx : idx])

        tmp_dict['spike_times'] = unit_spike_times

    # Other keys
    for key in group:
        if 'spike_times' in key: continue

        if isinstance(group[key][0], bytes):
            tmp_dict[key] = list(map(lambda x: x.decode(), group[key][:]))
        else:
            tmp_dict[key] = list(group[key][:])
        
    return pd.DataFrame(tmp_dict)