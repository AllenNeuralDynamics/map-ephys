#! /usr/bin/env python

import os
import logging
import re
import pathlib

from datetime import date, datetime
from collections import namedtuple

import scipy.io as spio
import numpy as np
import pandas as pd
import decimal
import warnings
import datajoint as dj
import json
import traceback

from pybpodgui_api.models.project import Project as BPodProject
from . import InvalidBehaviorTrialError
from .utils import foraging_bpod

from pipeline import lab, experiment
from pipeline import get_schema_name, dict_to_hash, create_schema_settings

schema = dj.schema(get_schema_name('ingest_behavior'), **create_schema_settings)

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

# ================ PHOTOSTIM PROTOCOL ===============
photostim_duration = 0.5  # (s)
skull_ref = 'Bregma'
photostims = {
    4: {'photo_stim': 4, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
        'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                       'ap_location': 2500, 'ml_location': -1500, 'depth': 0,
                       'theta': 15, 'phi': 15}]},
    5: {'photo_stim': 5, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
        'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                       'ap_location': 2500, 'ml_location': 1500, 'depth': 0,
                       'theta': 15, 'phi': 15}]},
    6: {'photo_stim': 6, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
        'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                       'ap_location': 2500, 'ml_location': -1500, 'depth': 0,
                       'theta': 15, 'phi': 15},
                      {'skull_reference': skull_ref, 'brain_area': 'ALM',
                       'ap_location': 2500, 'ml_location': 1500, 'depth': 0,
                       'theta': 15, 'phi': 15}
                      ]}}


def get_behavior_paths():
    '''
    retrieve behavior rig paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
          'behavior_data_paths':
            [
                ["RRig", "/path/string", 0],
                ["RRig2", "/path2/string2", 1]
            ],
        }
        ...
      }

    where 'behavior_data_paths' is a list of multiple possible path for behavior data, each in format:
    [rig name, rig full path, search order]
    '''

    paths = dj.config.get('custom', {}).get('behavior_data_paths', None)
    if paths is None:
        raise ValueError("Missing 'behavior_data_paths' in dj.config['custom']")

    return sorted(paths, key=lambda x: x[-1])


def get_session_user():
    '''
    Determine desired 'session user' for a session.

    - 1st, try dj.config['custom']['session.user']
    - 2nd, try dj.config['database.user']
    - else, use 'unknown'

    TODO: multi-user / bulk ingest support
    '''
    session_user = dj.config.get('custom', {}).get('session.user', None)

    session_user = (dj.config.get('database.user')
                    if not session_user else session_user)

    if len(lab.Person() & {'username': session_user}):
        return session_user
    else:
        return 'unknown'


@schema
class BehaviorIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> master
        behavior_file:              varchar(255)          # behavior file name
        """

    class CorrectedTrialEvents(dj.Part):
        ''' TrialEvents containing auto-corrected data '''
        definition = """
        -> BehaviorIngest
        -> experiment.TrialEvent
        """

    @property
    def key_source(self):

        # 2 letters, anything, _, anything, 8 digits, _, 6 digits, .mat
        # where:
        # (2 letters, anything): water restriction
        # (anything): task name
        # (8 digits): date YYYYMMDD
        # (6 digits): time HHMMSS

        rexp = '^[a-zA-Z]{2}.*_.*_[0-9]{8}_[0-9]{6}.mat$'

        # water_restriction_number -> subject
        h2os = {k: v for k, v in zip(*lab.WaterRestriction().fetch(
            'water_restriction_number', 'subject_id'))}

        def buildrec(rig, rigpath, root, f):

            if not re.match(rexp, f):
                log.debug("{f} skipped - didn't match rexp".format(f=f))
                return

            log.debug('found file {f}'.format(f=f))

            fullpath = pathlib.Path(root, f)
            subpath = fullpath.relative_to(rigpath)

            fsplit = subpath.stem.split('_')
            h2o = fsplit[0]
            ymd = fsplit[-2:-1][0]

            if h2o not in h2os:
                log.warning('{f} skipped - no animal for {h2o}'.format(
                    f=f, h2o=h2o))
                return

            animal = h2os[h2o]

            log.debug('animal is {animal}'.format(animal=animal))

            return {
                'subject_id': animal,
                'session_date': date(
                    int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])),
                'rig': rig,
                'rig_data_path': rigpath.as_posix(),
                'subpath': subpath.as_posix()
            }

        recs = []
        found = set()
        known = set(BehaviorIngest.BehaviorFile().fetch('behavior_file'))
        rigs = get_behavior_paths()

        for (rig, rigpath, _) in rigs:
            rigpath = pathlib.Path(rigpath)

            log.info('RigDataFile.make(): traversing {}'.format(rigpath))
            for root, dirs, files in os.walk(rigpath):
                log.debug('RigDataFile.make(): entering {}'.format(root))
                for f in files:
                    log.debug('RigDataFile.make(): visiting {}'.format(f))
                    r = buildrec(rig, rigpath, root, f)
                    if not r:
                        continue
                    if f in set.union(known, found):
                        log.info('skipping already ingested file {}'.format(
                            r['subpath']))
                    else:
                        found.add(f)  # block duplicate path conf
                        recs.append(r)

        return recs

    def populate(self, *args, **kwargs):
        # 'populate' which won't require upstream tables
        # 'reserve_jobs' not parallel, overloaded to mean "don't exit on error"
        for k in self.key_source:
            try:
                with dj.conn().transaction:
                    self.make(k)
            except Exception as e:
                log.warning('session key {} error: {}'.format(k, repr(e)))
                if not kwargs.get('reserve_jobs', False):
                    raise

    def make(self, key):
        log.info('BehaviorIngest.make(): key: {key}'.format(key=key))

        # File paths conform to the pattern:
        # dl7/TW_autoTrain/Session Data/dl7_TW_autoTrain_20180104_132813.mat
        # which is, more generally:
        # {h2o}/{training_protocol}/Session Data/{h2o}_{training protocol}_{YYYYMMDD}_{HHMMSS}.mat

        path = pathlib.Path(key['rig_data_path'], key['subpath'])

        # distinguishing "delay-response" task or "multi-target-licking" task
        task_type = detect_task_type(path)

        # skip too small behavior file (only for 'delay-response' task)
        if task_type == 'delay-response' and os.stat(path).st_size / 1024 < 1000:
            log.info('skipping file {} - too small'.format(path))
            return

        log.debug('loading file {}'.format(path))

        # Read from behavior file and parse all trial info (the heavy lifting here)
        skey, rows = BehaviorIngest._load(key, path, task_type)

        # Session Insertion

        log.info('BehaviorIngest.make(): adding session record')
        experiment.Session.insert1(skey)

        # Behavior Insertion

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): saving ingest {d}'.format(d=skey))
        self.insert1(skey, ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial.insert(
            rows['trial'], ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial.insert(
            rows['behavior_trial'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote.insert(
            rows['trial_note'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent.insert(
            rows['trial_event'], ignore_extra_fields=True,
            allow_direct_insert=True, skip_duplicates=True)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent.insert(
            rows['action_event'], ignore_extra_fields=True,
            allow_direct_insert=True)

        # Photostim Insertion

        photostim_ids = np.unique(
            [r['photo_stim'] for r in rows['photostim_trial_event']])

        unknown_photostims = np.setdiff1d(
            photostim_ids, list(photostims.keys()))

        if unknown_photostims:
            raise ValueError(
                'Unknown photostim protocol: {}'.format(unknown_photostims))

        if photostim_ids.size > 0:
            log.info('BehaviorIngest.make(): ... experiment.Photostim')
            for stim in photostim_ids:
                experiment.Photostim.insert1(
                    dict(skey, **photostims[stim]), ignore_extra_fields=True)

                experiment.Photostim.PhotostimLocation.insert(
                    (dict(skey, **loc,
                          photo_stim=photostims[stim]['photo_stim'])
                     for loc in photostims[stim]['locations']),
                    ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrial')
        experiment.PhotostimTrial.insert(rows['photostim_trial'],
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrialEvent')
        experiment.PhotostimEvent.insert(rows['photostim_trial_event'],
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

        if task_type == 'multi-target-licking':
            # Multi-target-licking specifics
            log.info('BehaviorIngest.make(): ... experiment.MultiTargetLickingSessionBlock')
            experiment.MultiTargetLickingSessionBlock.insert(
                rows['session_block'],
                ignore_extra_fields=True,
                allow_direct_insert=True)

            log.info('BehaviorIngest.make(): ... experiment.MultiTargetLickingSessionBlock.WaterPort')
            experiment.MultiTargetLickingSessionBlock.WaterPort.insert(
                rows['session_block_waterport'],
                ignore_extra_fields=True,
                allow_direct_insert=True)

            log.info('BehaviorIngest.make(): ... experiment.MultiTargetLickingSessionBlock.BlockTrial')
            experiment.MultiTargetLickingSessionBlock.BlockTrial.insert(
                rows['session_block_trial'],
                ignore_extra_fields=True,
                allow_direct_insert=True)

        # Behavior Ingest Insertion

        log.info('BehaviorIngest.make(): ... BehaviorIngest.BehaviorFile')
        BehaviorIngest.BehaviorFile.insert1(
            dict(skey, behavior_file=os.path.basename(key['subpath'])),
            ignore_extra_fields=True, allow_direct_insert=True)

    @classmethod
    def _load(cls, key, path, task_type):
        """
        Method to load the behavior file (.mat), parse trial info and prepare for insertion
        (no table insertion is done here)

        :param key: session_key
        :param path: (str) filepath of the behavior file (.mat)
        :param task_type: (str) "delay-response" or "multi-target-licking"
        :return: skey, rows
            + skey: session_key
            + rows: a dictionary containing all per-trial information to be inserted
        """
        path = pathlib.Path(path)

        h2o = (lab.WaterRestriction() & {'subject_id': key['subject_id']}).fetch1(
            'water_restriction_number')

        ymd = key['session_date']
        datestr = ymd.strftime('%Y%m%d')
        log.info('h2o: {h2o}, date: {d}'.format(h2o=h2o, d=datestr))

        # session key
        skey = {}
        skey['subject_id'] = key['subject_id']
        skey['session_date'] = ymd
        skey['username'] = get_session_user()
        skey['rig'] = key['rig']
        skey['h2o'] = h2o
        # synthesizing session ID
        log.debug('synthesizing session ID')
        session = (dj.U().aggr(experiment.Session()
                               & {'subject_id': skey['subject_id']},
                               n='max(session)').fetch1('n') or 0) + 1
        log.info('generated session id: {session}'.format(session=session))
        skey['session'] = session

        if task_type == 'multi-target-licking':
            rows = load_multi_target_licking_matfile(skey, path)
        elif task_type == 'delay-response':
            rows = load_delay_response_matfile(skey, path)
        else:
            raise ValueError('Unknown task-type: {}'.format(task_type))

        return skey, rows


@schema
class BehaviorBpodIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> master
        behavior_file:              varchar(255)          # behavior file name
        """

    water_port_name_mapper = {'left': 'L', 'right': 'R', 'middle': 'M'}

    @staticmethod
    def get_bpod_projects():
        projectdirs = dj.config.get('custom', {}).get('behavior_bpod', []).get('project_paths')
        # construct a list of BPod Projects
        projects = []
        for projectdir in projectdirs:
            projects.append(BPodProject())
            projects[-1].load(projectdir)
        return projects
    
    @staticmethod
    def _get_message(df, MSG):
        """
        Get message from df_behavior_trial
        Locate the row of 'MSG' and return the next content in the next row
        """
        index_msg = df['MSG'].str.contains(MSG) 
        if sum(index_msg == True):
            return df.loc[df.index[df['MSG'].str.contains(MSG)] + 1, 'MSG']
        else:
            return None


    @property
    def key_source(self):
        key_source = []

        IDs = {k: v for k, v in zip(*(lab.WaterRestriction & 
                                      '''water_restriction_number NOT LIKE "DL%" AND 
                                         water_restriction_number NOT LIKE "dl%" AND 
                                         water_restriction_number NOT LIKE "SC%" AND
                                         water_restriction_number NOT LIKE "tw%"'''
                                        #   & 'water_restriction_number = "XY_14"'
                                      ).fetch(
            'water_restriction_number', 'subject_id'))}

        for subject_now, subject_id_now in IDs.items():
            meta_dir = dj.config.get('custom', {}).get('behavior_bpod', []).get('meta_dir')

            subject_csv = pathlib.Path(meta_dir) / '{}.csv'.format(subject_now)
            if subject_csv.exists():
                df_wr = pd.read_csv(subject_csv)
            else:
                log.info('No metadata csv found for {}'.format(subject_now))
                continue

            for r_idx, df_wr_row in df_wr.iterrows():
                # we use it when both start and end times are filled in and Water during training > 0; restriction, freewater and handling is skipped
                if (df_wr_row['Time'] and isinstance(df_wr_row['Time'], str)
                        and df_wr_row['Time-end'] and isinstance(df_wr_row['Time-end'], str)
                        and 'foraging' in str(df_wr_row['Training type'])):

                    for f in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
                        try:
                            date_now = datetime.strptime(df_wr_row.Date.split(' ')[0], f).date()
                            break
                        except:
                            pass
                    else:
                        log.info('Unable to parse session date for {}: {}. Skipping...'.format(
                            subject_now, df_wr_row.Date))
                        continue

                    if not (experiment.Session & {'subject_id': subject_id_now,
                                                  'session_date': date_now}):
                        try:
                            session_number_xls = int(df_wr_row['Training Session']) if 'Training Session' in df_wr_row else None
                        except:
                            session_number_xls = None
                        
                        key_source.append({'subject_id': subject_id_now,
                                           'session_date': date_now,
                                           'session_comment': str(df_wr_row['Notes']),
                                           'session_weight': df_wr_row['Weight'],
                                           'session_water_earned': df_wr_row['Water during training'],
                                           'session_water_extra': df_wr_row['Extra water'],
                                           'session_number_xls': session_number_xls})

        return key_source

    def populate(self, *args, **kwargs):
        # Load project info (just once)
        log.info('------ Loading pybpod project -------')
        self.projects = self.get_bpod_projects()
        log.info('------------   Done! ----------------')

        # 'populate' which won't require upstream tables
        # 'reserve_jobs' not parallel, overloaded to mean "don't exit on error"                          
        for k in self.key_source:
            try:
                with dj.conn().transaction:
                    self.make(k)
            except Exception as e:
                log.warning('session key {} error: {}'.format(k, repr(e)))
                traceback.print_exc()
                if not kwargs.get('reserve_jobs', False):
                    raise

    def make(self, key):
        log.info(
            '----------------------\nBehaviorBpodIngest.make(): key: {key}'.format(key=key))

        subject_id_now = key['subject_id']
        subject_now = (lab.WaterRestriction() & {'subject_id': subject_id_now}).fetch1(
            'water_restriction_number')
        date_now_str = key['session_date'].strftime('%Y%m%d')
                        
        log.info('h2o: {h2o}, date: {d}'.format(h2o=subject_now, d=date_now_str))

        # ---- Ingest information for BPod projects ----
        sessions_now, session_start_times_now, experimentnames_now = [], [], []
        for proj in self.projects:  #
            exps = proj.experiments
            for exp in exps:
                stps = exp.setups
                for stp in stps:
                    for session in stp.sessions:
                        if (session.subjects and 
                            (session.subjects[0].find(subject_now) > -1 or
                             (subject_now == "XY_04" and session.subjects[0].find("XX_04") > -1))  # Exceptions                            
                                and session.name.startswith(date_now_str)):
                            sessions_now.append(session)
                            session_start_times_now.append(session.started)
                            experimentnames_now.append(exp.name)
                            
        bpodsess_order = np.argsort(session_start_times_now)

        # --- Handle missing BPod session ---
        if len(bpodsess_order) == 0:
            log.error('BPod session not found!')
            return

        # ---- Concatenate bpod sessions (and corresponding trials) into one datajoint session ----
        tbls_2_insert = ('sess_trial', 'behavior_trial', 'trial_note',
                         'sess_block', 'sess_block_trial',
                         'trial_choice', 'trial_event', 'action_event',
                         'photostim', 'photostim_location', 'photostim_trial',
                         'photostim_trial_event',
                         'valve_setting', 'valve_open_dur', 'available_reward',
                         'photostim_foraging_trial')

        # getting started
        concat_rows = {k: list() for k in tbls_2_insert}
        sess_key = None
        trial_num = 0  # trial numbering starts at 1

        for s_idx, session_idx in enumerate(bpodsess_order):
            session = sessions_now[session_idx]
            experiment_name = experimentnames_now[session_idx]
            csvfilename = (pathlib.Path(session.path) / (
                        pathlib.Path(session.path).name + '.csv'))

            # ---- Special parsing for csv file ----
            log.info('Load session file(s) ({}/{}): {}'.format(s_idx + 1, len(bpodsess_order),
                                                               csvfilename))
            df_behavior_session = foraging_bpod.load_and_parse_a_csv_file(csvfilename)

            # ---- Integrity check of the current bpodsess file ---
            # It must have at least one 'trial start' and 'trial end'
            trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (
                        df_behavior_session['MSG'] == 'New trial')].index
            
            if not len(trial_start_idxs):
                log.info('No "trial start" for {}. Skipping...'.format(csvfilename))
                continue  # Make sure 'start' exists, otherwise move on to try the next bpodsess file if exists

            trial_end_idxs = df_behavior_session[
                (df_behavior_session['TYPE'] == 'TRANSITION') & (
                            df_behavior_session['MSG'] == 'End')].index
            if not len(trial_end_idxs):
                log.info('No "trial end" for {}. Skipping...'.format(csvfilename))
                continue  # Make sure 'end' exists, otherwise move on to try the next bpodsess file if exists

            # It must be a foraging session
            # extracting task protocol - hard-code implementation
            if 'foraging_randomwalk' in experiment_name.lower():
                task = 'foraging'
                task_protocol = 120
                lick_ports = ['left', 'right']
            elif 'foraging_uncoupled' in experiment_name.lower():
                task = 'foraging'
                task_protocol = 110
                lick_ports = ['left', 'right']
            elif 'foraging' in experiment_name.lower() or (
                    'bari' in experiment_name.lower() and 'cohen' in experiment_name.lower()):
                if 'var:lickport_number' in df_behavior_session and \
                        df_behavior_session['var:lickport_number'][0] == 3:
                    task = 'foraging 3lp'
                    task_protocol = 101
                    lick_ports = ['left', 'right', 'middle']
                else:
                    task = 'foraging'
                    task_protocol = 100
                    lick_ports = ['left', 'right']
            else:
                log.info('ERROR: unhandled task name {}. Skipping...'.format(experiment_name))
                continue  # Make sure this is a foraging bpodsess, otherwise move on to try the next bpodsess file if exists

            # ---- New session - construct a session key (from the first bpodsess that passes the integrity check) ----
            if sess_key is None:
                session_time = df_behavior_session['PC-TIME'][trial_start_idxs[0]]
                if session.setup_name.lower() in ['day1', 'tower-2', 'day2-7', 'day_1',
                                                  'real foraging']:
                    setupname = 'AIND-Tower-2' if key['session_date'] > date(2022, 1, 1) else 'Training-Tower-2'
                elif session.setup_name.lower() in ['tower-3', 'tower-3beh', ' tower-3', '+',
                                                    'tower 3']:
                    setupname = 'AIND-Tower-3' if key['session_date'] > date(2022, 1, 1) else 'Training-Tower-3'
                elif session.setup_name.lower() in ['tower-1']:
                    setupname = 'AIND-Tower-1' if key['session_date'] > date(2022, 1, 1) else 'Training-Tower-1'
                elif session.setup_name.lower() in ['tower-4']:
                    setupname = 'Training-Tower-4'
                elif session.setup_name.lower() in ['ephys_han']:
                    setupname = 'AIND-Ephys-Han' if key['session_date'] > date(2022, 1, 1) else 'Ephys-Han'
                elif 'aind-tower' in session.setup_name.lower():
                    setupname = session.setup_name  # Correctly formated AIND tower. Such as 'AIND-Tower-4'
                else:
                    log.info('ERROR: unhandled setup name {} (from {}). Skipping...'.format(
                        session.setup_name, session.path))
                    continue  # Another integrity check here
                
                log.debug('synthesizing session ID')
                synthesized_session_ID = (dj.U().aggr(experiment.Session()
                                              & {'subject_id': subject_id_now},
                                              n='max(session)').fetch1('n') or 0) + 1
                
                if key['session_number_xls'] is not None:
                    key['session'] = key['session_number_xls']
                    if key['session_number_xls'] != synthesized_session_ID:
                        log.info(f"WARNING: ingested using session ID = {key['session_number_xls']}, but synthesized session ID = {synthesized_session_ID}!")
                    else:
                        log.info(f"session ID = {key['session_number_xls']} (matched)")
                else:
                    key['session'] = synthesized_session_ID  # Old method
                    log.info(f"session ID = {synthesized_session_ID} (synthesized)")
                
                sess_key = {**key,
                            'session_time': session_time.time(),
                            'username': df_behavior_session['experimenter'][0],
                            'rig': setupname}

            # ---- channel for water ports ----
            water_port_channels = {}
            for lick_port in lick_ports:
                chn_varname = 'var:WaterPort_{}_ch_in'.format(
                    self.water_port_name_mapper[lick_port])
                if chn_varname not in df_behavior_session:
                    log.error(
                        'Bpod CSV KeyError: {} - Available columns: {}'.format(chn_varname,
                                                                               df_behavior_session.columns))
                    return
                water_port_channels[lick_port] = df_behavior_session[chn_varname][0]

            # ---- Ingestion of trials ----

            # extracting trial data
            session_start_time = datetime.combine(sess_key['session_date'], sess_key['session_time'])
            
            if sess_key['session_date'] > date(2021, 5, 23):
                # starting from https://github.com/hanhou/Foraging-Pybpod/commit/775dff4435a8c9b8ad14501f0b44de741d683737
                trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'stdout') & (df_behavior_session['MSG'] == 'Trialnumber:')].index - 2
            else:
                trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
                trial_start_idxs -= 2 # To reflect the change that bitcode is moved before the "New trial" line
            
            trial_start_idxs = pd.Index([0]).append(trial_start_idxs[1:])  # so the random seed will be present
            trial_end_idxs = trial_start_idxs[1:].append(pd.Index([(max(df_behavior_session.index))]))

            # trial_end_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'END-TRIAL')].index
            prevtrialstarttime = np.nan
            blocknum_local_prev = np.nan
            last_iti_after_on_PC_time, last_iti_after_down_PC_time, last_iti_after_off_PC_time = None, None, None

            # getting ready
            rows = {k: list() for k in
                    tbls_2_insert}  # lists of various records for batch-insert

            for trial_start_idx, trial_end_idx in zip(trial_start_idxs, trial_end_idxs):
                df_behavior_trial = df_behavior_session[trial_start_idx:trial_end_idx + 1]

                # Trials without GoCue are skipped
                if not len(
                        df_behavior_trial[(df_behavior_trial['MSG'] == 'GoCue') & (
                        df_behavior_trial['TYPE'] == 'STATE')]):
                    continue

                # ---- session trial ----
                trial_num += 1  # increment trial number
                trial_uid = len(experiment.SessionTrial & {'subject_id': subject_id_now}) + trial_num  # Fix trial_uid here
                
                # Note that the following trial_start/stop_time SHOULD NEVER BE USED in ephys related analysis 
                # because they are PC-TIME, which is not accurate (4 ms average delay, sometimes up to several seconds!)!
                # In fact, from bpod.csv, we can only accurately retrieve (local) trial-wise, but not (global) session-wise, times
                # See comments below.
                trial_start_time = df_behavior_session['PC-TIME'][
                                       trial_start_idx].to_pydatetime() - session_start_time
                trial_stop_time = df_behavior_session['PC-TIME'][
                                      trial_end_idx].to_pydatetime() - session_start_time

                sess_trial_key = {**sess_key,
                                  'trial': trial_num,
                                  'trial_uid': trial_uid,
                                  'start_time': trial_start_time.total_seconds(),
                                  'stop_time': trial_stop_time.total_seconds()}
                rows['sess_trial'].append(sess_trial_key)

                # ---- session block ----
                if task_protocol in (100,): # 'Block_number' in df_behavior_session:
                    if np.isnan(df_behavior_trial['Block_number'].to_list()[0]):
                        blocknum_local = 0 if np.isnan(
                            blocknum_local_prev) else blocknum_local_prev
                    else:
                        blocknum_local = int(
                            df_behavior_trial['Block_number'].to_list()[0]) - 1
                        blocknum_local_prev = blocknum_local

                    reward_probability = {}
                    for lick_port in lick_ports:
                        p_reward_varname = 'var:reward_probabilities_{}'.format(
                            self.water_port_name_mapper[lick_port])
                        reward_probability[lick_port] = decimal.Decimal(
                            df_behavior_session[p_reward_varname][0][blocknum_local]).quantize(
                            decimal.Decimal(
                                '.001'))  # Note: Reward probabilities never changes during a **bpod** session
                            
                elif task_protocol in (110, 120):  # Decoupled blocks and Random walk: reward probs were generated on-the-fly
                    p_L = json.loads(self._get_message(df_behavior_trial, 'reward_p_L').iloc[0])
                    p_R = json.loads(self._get_message(df_behavior_trial, 'reward_p_R').iloc[0])
                    reward_probability = {'left': p_L, 'right': p_R}
                

                # determine if this is a new block: compare reward probability with the previous block
                if rows['sess_block']:
                    itsanewblock = dict_to_hash(reward_probability) != dict_to_hash(
                        rows['sess_block'][-1]['reward_probability'])
                else:
                    itsanewblock = True

                if itsanewblock:
                    all_blocks = [b['block'] for b in
                                    rows['sess_block'] + concat_rows['sess_block']]
                    block_num = (np.max(all_blocks) + 1 if all_blocks else 1)
                    rows['sess_block'].append({**sess_key,
                                                'block': block_num,
                                                'block_start_time': trial_start_time.total_seconds(),
                                                'reward_probability': reward_probability})
                else:
                    block_num = rows['sess_block'][-1]['block']

                rows['sess_block_trial'].append({**sess_trial_key, 'block': block_num})


                # ====== Event times ======
                # Foraging trial structure: (*...*: events of interest in experiment.EventType; [...]: optional)
                # -> (ITI) -> *bitcodestart* -> bitcode -> lickport movement -> *delay* (lickport in position) 
                #          -> [effective delay period] -> *go* -> [*choice*] -> [*reward*] -> *trialend* -> (ITI) ->
                # Notes:
                # 1. Differs from the delay-task:
                #       (1) no sample and presample epoch
                #       (2) effective delay period could be zero (ITI as an inherent delay). 
                #           Also note that if the delay_period in bpod protocol < lickport movement time (~100 ms), the effective delay period is also zero, 
                #           where the go-cue sound actually appears BEFORE the lickport stops moving.
                #       (3) we are interested in not only go-cue aligned PSTH (ephys.Unit.TrialSpikes), but need more flexible event alignments, especially ITI firings.
                #           So we should use the session-wise untrialized spike times stored in ephys.Unit['spike_times']. See below.
                # 2. Two "trial start"s:
                #       (1) *bitcodestart* = onset of the first bitcode = *sTrig* in NIDQ bitcode.mat
                #       (2) `trial_start_idx` in this for loop = the start of bpod-trial ('New Trial' in bpod csv file)
                #            = the reference point of BPOD-TIME  = NIDQ bpod-trial channel
                #    They are far from each other because I start the bpod trial at the middle of ITI (Foraging_bpod: e9a8ffd6) to cover video recording during ITI.
                #    --> Update: now I move the bpod trial at the end of the last trial!! (ITI after = 0), starting from Foraging_bpod: f86af9e901, 2022-05-15
                # 3. In theory, *bitcodestart* = *delay* (since there's no sample period),
                #    but in practice, the bitcode (21*20=420 ms) and lickport movement (~100 ms) also take some time.
                #    Note that bpod doesn't know the exact time when lickports are in place, so we can get *delay* only from NIDQ zaber channel (ephys.TrialEvent.'zaberinposition').
                # 4. In early lick trials, effective delay start should be the last 'DelayStart' (?)
                # 5. Finally, to perform session-wise alignment between behavior and ephys, there are two ways, which could be cross-checked with each other:
                #       (1) (most straightforward) use all event markers directly from NIDQ bitcode.mat,
                #           then align them to ephys.Unit['spike_times'] by looking at the *sTrig* of the first trial of a session
                #       (2) (can be used as a sanity check) extract trial-wise BPOD-TIME from pybpod.csv,
                #           and then convert the local trial-wise times to global session-wise times by aligning
                #           the same events from pybpod.csv and bitcode.mat across all trials, e.g., *bitcodestart* <--> *sTrig*, or 0 <--> NIDQ bpod-"trial trigger" channel
                #    Note that one should NEVER use PC-TIME from the bpod csv files (at least for ephys-related alignment)!!!
                
                # ----- BPOD STATES (all events except licks and photostim) -----
                bpod_states_this_trial = df_behavior_trial[(df_behavior_trial['TYPE'] == 'STATE') & (df_behavior_trial['BPOD-INITIAL-TIME'] > 0)]   # All states of this trial
                trial_event_count = 0

                # Use BPOD-INITIAL-TIME and BPOD-FINAL-TIME (all relative to bpod-trialstart)
                bpod_states_of_interest = { # experiment.TrialEventType: Bpod state name
                                           'videostart': ['ITIBeforeVideoOn'],
                                           'bitcodestart': ['Start'],
                                           'delay': ['DelayStart'],     # (1) in a non early lick trial, effective delay start = max(DelayStart, LickportInPosition).
                                                                        #       where LickportIntInPosition is only available from NIDQ
                                                                        # (2) in an early lick trial, there are multiple DelayStarts, the last of which is the effective delay start
                                           'go': ['GoCue'],
                                           'choice': [f'Choice_{lickport}' for lickport in self.water_port_name_mapper.values()],
                                           'reward': [f'Reward_{lickport}' for lickport in self.water_port_name_mapper.values()],
                                           'doubledip': ['Double_dipped'] + [f'Double_dipped_to_{lickport}' for lickport in self.water_port_name_mapper.values()],   # Only for non-double-dipped trials, ITI = last lick + 1 sec (maybe I should not use double dipping punishment for ehpys?)
                                           'trialend': ['ITI'],
                                           'videoend': ['ITIAfterVideoOff'],
                                           }

                for trial_event_type, bpod_state in bpod_states_of_interest.items():
                    _idx = bpod_states_this_trial.index[bpod_states_this_trial['MSG'].isin(bpod_state)]   # One state could have multiple appearances, such as DelayStart in early-lick trials
                    if not len(_idx):
                        continue

                    initials, finals = bpod_states_this_trial.loc[_idx][['BPOD-INITIAL-TIME', 'BPOD-FINAL-TIME']].values.T.astype(float)
                    initials[initials > 9999] = 9999   # Wordaround for bug #9: BPod protocol was paused and then resumed after an impossible long period of time (> decimal(8, 4)).
                    finals[finals > 9999] = 9999

                    # cache event times
                    for idx, (initial, final) in enumerate(zip(initials, finals)):
                        rows['trial_event'].extend(
                            [{**sess_trial_key,
                              'trial_event_id': trial_event_count + idx,
                              'trial_event_type': trial_event_type,
                              'trial_event_time': initial,
                              'duration': final - initial}])   # list comprehension doesn't work here

                    trial_event_count += len(initials)
                    
                    # save gocue time for early-lick below
                    if trial_event_type == 'go':
                        gocue_time = initials[0]
                        
                # ------ Photostim info from csv (sorry for this very complicated logic to capture different versions of bpod protocol...) -------
                # Act as a master switch:
                laser_power_txt = self._get_message(df_behavior_trial, 'laser power')
                
                # if_laser_this_trial = len(laser_power_txt) # TODO make this compatible before 2021-08-25 
                timer4_start = df_behavior_trial.index[df_behavior_trial['+INFO'] == 'GlobalTimer4_Start']
                timer5_start = df_behavior_trial.index[df_behavior_trial['+INFO'] == 'GlobalTimer5_Start']
                photostim_event_id = 0
                
                if len(timer4_start) or len(timer5_start):
                    # Retrieve laser power
                    laser_power_list = json.loads(laser_power_txt.iloc[0])  # Assuming both sides have the same power (calibrated)
                    if len(laser_power_list) == 3:
                        laser_power = laser_power_list[0]  # Calibrated laser power
                    elif len(laser_power_list) == 2:   # Use post-hoc calibration
                        laser_cali =  np.array([
                            [0.5, 0.14, 0.08],
                            [1.0, 0.25, 0.15],
                            [2.5, 0.75, 0.55],
                            [5.0, 1.5, 1.1],
                            [7.5, 2.2, 1.65],
                            [10.0, 2.95, 2.25],
                            [13.5, 4.8, 3.7]     
                        ])
                        p = laser_cali[:, 0]
                        v = np.mean(laser_cali[:, 1:], axis=1) 
                        laser_power = np.interp(laser_power_list[1], v, p)   # Approximated power
                        
                    # Retrieve WavePlayer events
                    side_event_mapping = {'L': ['WavePlayer1_2', 'WavePlayer1_6'], 'R': ['WavePlayer1_4', 'WavePlayer1_6']}
                    stim_sides = ''
                    time_saved = False
                    
                    for side, event in side_event_mapping.items():
                        wavplayer_rows = df_behavior_trial.loc[df_behavior_trial['+INFO'].isin(event), 'BPOD-INITIAL-TIME']
                        
                        if len(wavplayer_rows): # laser on, laser down, laser off
                            stim_sides += side

                            if not time_saved:  # Only save time once (here we assume if both sides are stimulated, they are in sync)
                                time_saved = True
                                
                                # Decode on_bpod_time, down_bpod_time, and off_bpod_time of this *Photostim* trial (not bpod trial for the old protocol)
                                if key['session_date'] >= date(2022, 5, 15):  # New protocol: bpod trial starts at the end of the last trial
                                    timer4_start_bpod_time = df_behavior_trial.loc[timer4_start, 'BPOD-INITIAL-TIME']
                                    if len(wavplayer_rows) == 2:   
                                        if wavplayer_rows.iloc[1] - timer4_start_bpod_time.iloc[0] > 0.001:  # To capture the bug of missing first 'WavePlayer1_x' after timer4_start
                                            wavplayer_rows = timer4_start_bpod_time.append(wavplayer_rows)  # Insert the fake first `WavePlayer1_x` using timer4_start
                                            log.info(f"Warning: , len({event}) = 2 @ {subject_now}, session {key['session']}, trial {trial_num}. Fixed!")
                                        else:
                                            log.info(f"Warning: , len({event}) = 2 @ {subject_now}, session {key['session']}, trial {trial_num}. Cannot fix!")
                                            breakpoint()
                                    elif len(wavplayer_rows) == 1:  # In case where photostim starts after go cue and terminates at the end of the trial (no active stop or ramping down)
                                        if wavplayer_rows.iloc[0] - timer4_start_bpod_time.iloc[0] < 0.001:  # The only `WavePlayer` event should be very close to TimerStart
                                            trial_end = df_behavior_trial.loc[df_behavior_trial['MSG'] == 'End', 'BPOD-INITIAL-TIME'].iloc[0]
                                            wavplayer_rows = wavplayer_rows.append(pd.Series([trial_end, trial_end]))  # photosim terminates at trial end; no ramping down
                                        else:
                                            log.info(f"Warning: , len({event}) = 1 @ {subject_now}, session {key['session']}, trial {trial_num}. Cannot fix!")
                                            breakpoint()
                                    on_bpod_time, down_bpod_time, off_bpod_time = wavplayer_rows.iloc[0:3]

                                elif key['session_date'] < date(2022, 5, 15):  # Old protocol: bpod trial starts at the middle of ITI
                                    this_bpod_start_PC_time = df_behavior_trial.loc[df_behavior_trial['MSG'] == 'New trial', 'PC-TIME']

                                    if len(timer4_start):   # The first three event_row should be this ITI before
                                        if len(wavplayer_rows) < 3: 
                                            print(f'Warning: has timer4 but len(waveplayer_row) < 3!!')
                                            breakpoint()
                                            continue
                                        if (last_iti_after_on_PC_time is not None and
                                            last_iti_after_off_PC_time is None and
                                            wavplayer_rows.iloc[0] < 0.001):  
                                            # laser of the last bpod trial extends to the start of this bpod trial!
                                            on_bpod_time = (last_iti_after_on_PC_time.iloc[0] - this_bpod_start_PC_time.iloc[0]).total_seconds()
                                        else:
                                            on_bpod_time = wavplayer_rows.iloc[0]
                                        down_bpod_time, off_bpod_time = wavplayer_rows.iloc[1], wavplayer_rows.iloc[2] # This should be always true
                                        
                                    if not len(timer4_start) and last_iti_after_on_PC_time is not None:  # Only "ITI after" (first half ITI) is stimulated
                                        on_bpod_time =   (last_iti_after_on_PC_time.iloc[0] - this_bpod_start_PC_time.iloc[0]).total_seconds()
                                        if last_iti_after_off_PC_time is not None:   # ramping down in the last bpod trial
                                            down_bpod_time = (last_iti_after_down_PC_time - this_bpod_start_PC_time.iloc[0]).total_seconds()
                                            off_bpod_time =  (last_iti_after_off_PC_time - this_bpod_start_PC_time.iloc[0]).total_seconds()
                                        else:  # No ramping down, hard stop at the start of this trial
                                            down_bpod_time, off_bpod_time = 0, 0
                                            
                                    # Cache for the next photostim trial. This is based on PC-time (less accurate than bpod time)
                                    # So far, ignore the case where there is a gap in the middle of ITI... (which should be very rare)
                                    if len(timer5_start):
                                        wavplayer_after_timer5 = wavplayer_rows[wavplayer_rows > df_behavior_trial.loc[timer5_start, 'BPOD-INITIAL-TIME'].values[0]]
                                        last_iti_after_on_PC_time = df_behavior_trial.loc[timer5_start, 'PC-TIME']

                                        if len(wavplayer_after_timer5) == 1:   # This ITIafter, laser ends until trial end (no ramping down)
                                            last_iti_after_down_PC_time = None
                                            last_iti_after_off_PC_time = None
                                        elif len(wavplayer_after_timer5) == 3:   # In this ITIafter, laser ends before trial end (ramping down; if timer4_start in next trial, ignored this ITI after)
                                            last_iti_after_down_PC_time = df_behavior_trial.loc[wavplayer_after_timer5.index[1], 'PC-TIME']
                                            last_iti_after_off_PC_time  = df_behavior_trial.loc[wavplayer_after_timer5.index[2], 'PC-TIME']
                                        else:
                                            print(f'ERROR: has timer5 but len(waveplayer_row after timer 5) is not 1 or 3!!')
                                            breakpoint()
                                    else:
                                        last_iti_after_on_PC_time = None
                                        last_iti_after_down_PC_time = None
                                        last_iti_after_off_PC_time = None
                                        
                                # Compute time to go cue
                                on_to_go_cue = on_bpod_time - gocue_time
                                off_to_go_cue = off_bpod_time - gocue_time
                                duration = off_to_go_cue - on_to_go_cue
                                ramping_down = round(off_bpod_time - down_bpod_time, 2)  # To the precision of 10 ms 
                                
                            # Fill in TrialEventTypes: 'laserLon', 'laserLdown', 'laserLoff', 'laserRon', 'laserRdown', 'laserRoff'
                            for bpod_time, suffix in zip([on_bpod_time, down_bpod_time, off_bpod_time], ['on', 'down', 'off']):
                                rows['trial_event'].extend(
                                    [{**sess_trial_key,
                                    'trial_event_id': trial_event_count,
                                    'trial_event_type': f'laser{side}{suffix}',
                                    'trial_event_time': bpod_time,
                                    'duration': 0}])
                                trial_event_count += 1
                                                                
                    # For experiment.PhotostimForaging
                    if stim_sides == '': 
                        log.info(f"WARNING: stim_sides = '' @ {subject_now}, session {key['session']}, trial {trial_num}!")
                        continue
                    
                    side_code = {'L': 0, 'R': 1, 'LR': 2, 'RL': 2}[stim_sides]
                    this_row = {**sess_trial_key,
                                'photostim_event_id': photostim_event_id,  # dummy for now. just in case there are multiple laser events for one trial
                                'side': side_code,
                                'power': laser_power,
                                'on_to_go_cue': on_to_go_cue,
                                'off_to_go_cue': off_to_go_cue,
                                'duration': duration,
                                'ramping_down': ramping_down,
                                }
                    photostim_event_id += 1   
                    
                    # Nullables and sanity checks
                    align_to_from_stdout = self._get_message(df_behavior_trial, 'laser aligned to')
                    if len(align_to_from_stdout):
                        timer_duration = json.loads(self._get_message(df_behavior_trial, 'laser timer duration').iloc[0])
                        if type(timer_duration) == list:
                            duration = timer_duration[0]
                        else:
                            duration = timer_duration
                            
                        if int(duration) == 1000 and align_to_from_stdout.values[0].lower() == 'After ITI START'.lower():  # New "whole trial" condition
                            this_row['bpod_timer_align_to'] = 'whole trial'
                        else:
                            this_row['bpod_timer_align_to'] = align_to_from_stdout.values[0].lower()
                        
                    timer_offset_from_stdout = self._get_message(df_behavior_trial, 'laser timer offset')
                    if len(timer_offset_from_stdout):
                        try:
                            tmp = json.loads(timer_offset_from_stdout.iloc[0])
                        except:
                            tmp = float(timer_offset_from_stdout.iloc[0].strip('[]'))
                        this_row['bpod_timer_offset'] = tmp[0] if isinstance(tmp, list) else tmp

                    side_code_from_stdout = self._get_message(df_behavior_trial, 'laser side')
                    if len(side_code_from_stdout):
                        assert side_code == side_code_from_stdout.astype(int).iloc[0], 'ERROR: stim_sides from WavePlayer != side from stdout message'
                        
                    ramping_down_from_stdout = self._get_message(df_behavior_session, 'laser ramping down')
                    if len(ramping_down_from_stdout):
                        if this_row['bpod_timer_align_to'] not in ('whole trial', 'after go cue'):  # Otherwise it's a hard stop
                            assert ramping_down == float(ramping_down_from_stdout.iloc[0]), 'ERROR: ramping down not consistent!!'
                    
                    rows['photostim_foraging_trial'].extend([this_row])
                
                else:  # if not (len(timer4_start) or len(timer5_start)):
                    # Set last trial laser to None to avoid wrongly decoded very long last ITI laser start
                    # (Last photostim trial in the old protocol is only "first half ITI stim", which is ignored here...)
                    last_iti_after_on_PC_time, last_iti_after_down_PC_time, last_iti_after_off_PC_time = None, None, None
                    
                # ------ Licks (use EVENT instead of STATE because not all licks triggered a state change) -------
                lick_times = {}
                for lick_port in lick_ports:
                    lick_times[lick_port] = df_behavior_trial['BPOD-INITIAL-TIME'][(
                                df_behavior_trial['+INFO'] == water_port_channels[lick_port])].to_numpy()
                    
                # cache licks
                all_lick_types = np.concatenate(
                    [[ltype] * len(ltimes) for ltype, ltimes in lick_times.items()])

                all_lick_times = np.concatenate(
                    [ltimes for ltimes in lick_times.values()])

                # sort by lick times
                sorted_licks = sorted(zip(all_lick_types, all_lick_times), key=lambda x: x[-1])

                rows['action_event'].extend([{**sess_trial_key, 'action_event_id': idx,
                                              'action_event_type': '{} lick'.format(ltype),
                                              'action_event_time': ltime} for
                                             idx, (ltype, ltime)
                                            in enumerate(sorted_licks)])

                # ====== Trial facts (nontemporal) ======
                # WaterPort Choice 
                trial_choice = {'water_port': None}
                for lick_port in lick_ports:
                    if any((df_behavior_trial['MSG'] == 'Choice_{}'.format(
                            self.water_port_name_mapper[lick_port]))
                           & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                        trial_choice['water_port'] = lick_port
                        break

                rows['trial_choice'].append({**sess_trial_key, **trial_choice})

                # early lick
                early_lick = 'no early'
                if any(all_lick_times < gocue_time):
                    early_lick = 'early'

                # outcome
                outcome = 'miss' if trial_choice['water_port'] else 'ignore'
                for lick_port in lick_ports:
                    if any((df_behavior_trial['MSG'] == 'Reward_{}'.format(
                            self.water_port_name_mapper[lick_port]))
                           & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                        outcome = 'hit'
                        break

                # ---- accumulated reward ----
                for lick_port in lick_ports:
                    reward_var_name = 'reward_{}_accumulated'.format(
                        self.water_port_name_mapper[lick_port])
                    if reward_var_name not in df_behavior_trial:
                        log.error('Bpod CSV KeyError: {} - Available columns: {}'.format(
                            reward_var_name, df_behavior_trial.columns))
                        return

                    reward = df_behavior_trial[reward_var_name].values[0]
                    rows['available_reward'].append({
                        **sess_trial_key, 'water_port': lick_port,
                        'reward_available': False if np.isnan(reward) else reward})

                # ---- auto water and notes ----
                auto_water = False
                auto_water_times = {}
                for lick_port in lick_ports:
                    auto_water_varname = 'Auto_Water_{}'.format(
                        self.water_port_name_mapper[lick_port])
                    auto_water_ind = (df_behavior_trial['TYPE'] == 'STATE') & (
                                df_behavior_trial['MSG'] == auto_water_varname)
                    if any(auto_water_ind):
                        auto_water = True
                        auto_water_times[lick_port] = float(
                            df_behavior_trial['+INFO'][auto_water_ind.idxmax()])

                if auto_water_times:
                    auto_water_ports = [k for k, v in auto_water_times.items() if v > 0.001]
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'autowater',
                                               'trial_note': 'and '.join(auto_water_ports)})

                # add random seed start note
                if any(df_behavior_trial['MSG'] == 'Random seed:'):
                    seedidx = (df_behavior_trial['MSG'] == 'Random seed:').idxmax() + 1
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'random_seed_start',
                                               'trial_note': str(df_behavior_trial['MSG'][seedidx])})
                    
                # add randomID (TrialBitCode)
                if any(df_behavior_trial['MSG'] == 'TrialBitCode: '):
                    bitcode_ind = (df_behavior_trial['MSG'] == 'TrialBitCode: ').idxmax() + 1
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'bitcode',
                                               'trial_note': str(df_behavior_trial['MSG'][bitcode_ind])})


                # ---- Behavior Trial ----
                rows['behavior_trial'].append({**sess_trial_key,
                                               'task': task,
                                               'task_protocol': task_protocol,
                                               'trial_instruction': 'none',
                                               'early_lick': early_lick,
                                               'outcome': outcome,
                                               'auto_water': auto_water,
                                               'free_water': False})  # TODO: verify this

                # ---- Water Valve Setting ----
                valve_setting = {**sess_trial_key}  

                if 'var_motor:LickPort_Lateral_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_lateral_pos'] = \
                        df_behavior_trial['var_motor:LickPort_Lateral_pos'].values[0]
                if 'var_motor:LickPort_RostroCaudal_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_rostrocaudal_pos'] = \
                        df_behavior_trial['var_motor:LickPort_RostroCaudal_pos'].values[0]
                if 'var_motor:LickPort_DorsoVentral_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_dorsoventral_pos'] = \
                        df_behavior_trial['var_motor:LickPort_DorsoVentral_pos'].values[0]

                rows['valve_setting'].append(valve_setting)

                for lick_port in lick_ports:
                    valve_open_varname = 'var:ValveOpenTime_{}'.format(
                        self.water_port_name_mapper[lick_port])
                    if valve_open_varname in df_behavior_trial:
                        rows['valve_open_dur'].append({
                            **sess_trial_key, 'water_port': lick_port,
                            'open_duration': df_behavior_trial[valve_open_varname].values[0]})

            # add to the session-concat
            for tbl in tbls_2_insert:
                concat_rows[tbl].extend(rows[tbl])
                
        # ---- The insertions to relevant tables ----
        # Session, SessionComment, SessionDetails insert
        log.info('BehaviorIngest.make(): adding session record')
        experiment.Session.insert1(sess_key, ignore_extra_fields=True)
        experiment.SessionComment.insert1(sess_key, ignore_extra_fields=True)
        experiment.SessionDetails.insert1(sess_key, ignore_extra_fields=True)

        # Behavior Insertion
        insert_settings = {'ignore_extra_fields': True, 'allow_direct_insert': True}

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial.insert(concat_rows['sess_trial'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial.insert(concat_rows['behavior_trial'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.WaterPortChoice')
        experiment.WaterPortChoice.insert(concat_rows['trial_choice'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote.insert(concat_rows['trial_note'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent.insert(concat_rows['trial_event'], **insert_settings)
        
        log.info('BehaviorIngest.make(): ... experiment.PhotostimForagingTrial')
        experiment.PhotostimForagingTrial.insert(concat_rows['photostim_foraging_trial'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent.insert(concat_rows['action_event'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.SessionBlock')
        experiment.SessionBlock.insert(concat_rows['sess_block'], **insert_settings)
        experiment.SessionBlock.BlockTrial.insert(concat_rows['sess_block_trial'],
                                                  **insert_settings)
        block_reward_prob = []
        for block in concat_rows['sess_block']:
            block_reward_prob.extend(
                [{**block, 'water_port': water_port, 'reward_probability': reward_p}
                 for water_port, reward_p in block.pop('reward_probability').items()])
        experiment.SessionBlock.WaterPortRewardProbability.insert(block_reward_prob,
                                                                  **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialAvailableReward')
        experiment.TrialAvailableReward.insert(concat_rows['available_reward'],
                                               **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.WaterValveSetting')
        experiment.WaterPortSetting.insert(concat_rows['valve_setting'], **insert_settings)
        experiment.WaterPortSetting.OpenDuration.insert(concat_rows['valve_open_dur'],
                                                        **insert_settings)

        # Behavior Ingest Insertion
        log.info('BehaviorBpodIngest.make(): saving ingest {}'.format(sess_key))
        self.insert1(sess_key, **insert_settings)
        self.BehaviorFile.insert(
            [{**sess_key, 'behavior_file': pathlib.Path(s.path).as_posix()}
             for s in sessions_now], **insert_settings)


# --------------------- HELPER LOADER FUNCTIONS -----------------

def detect_task_type(path):
    """
    Method to detect if a behavior matlab file is for "delay-response" or "multi-target-licking" task
    :param path: (str) filepath of the behavior file (.mat)
    :return task_type: (str) "delay-response" or "multi-target-licking"
    """
    # distinguishing "delay-response" task or "multi-target-licking" task
    mat = spio.loadmat(path.as_posix(), squeeze_me=True, struct_as_record=False)
    GUI_fields = set(mat['SessionData'].SettingsFile.GUI._fieldnames)

    if ({'X_center', 'Y_center', 'Z_center'}.issubset(GUI_fields)
            and not {'SamplePeriod', 'DelayPeriod'}.issubset(GUI_fields)):
        task_type = 'multi-target-licking'
    else:
        task_type = 'delay-response'

    return task_type


def load_delay_response_matfile(skey, matlab_filepath):
    """
    Loading routine for delay-response task - from .mat behavior data
    :param skey: session_key
    :param matlab_filepath: full-path to the .mat file containing the delay-response behavior data
    :return: nested list of all rows to be inserted into the various experiment-related tables
    """
    matlab_filepath = pathlib.Path(matlab_filepath)
    h2o = skey.pop('h2o')

    SessionData = spio.loadmat(matlab_filepath.as_posix(),
                               squeeze_me=True, struct_as_record=False)['SessionData']

    # parse session datetime
    session_datetime_str = str('').join((str(SessionData.Info.SessionDate), ' ',
                                         str(SessionData.Info.SessionStartTime_UTC)))
    session_datetime = datetime.strptime(
        session_datetime_str, '%d-%b-%Y %H:%M:%S')

    AllTrialTypes = SessionData.TrialTypes
    AllTrialSettings = SessionData.TrialSettings
    AllTrialStarts = SessionData.TrialStartTimestamp
    AllTrialStarts = AllTrialStarts - AllTrialStarts[0]  # real 1st trial

    RawData = SessionData.RawData
    AllStateNames = RawData.OriginalStateNamesByNumber
    AllStateData = RawData.OriginalStateData
    AllEventData = RawData.OriginalEventData
    AllStateTimestamps = RawData.OriginalStateTimestamps
    AllEventTimestamps = RawData.OriginalEventTimestamps

    AllRawEvents = SessionData.RawEvents.Trial

    # verify trial-related data arrays are all same length
    assert (all((x.shape[0] == AllStateTimestamps.shape[0] for x in
                 (AllTrialTypes, AllTrialSettings,
                  AllStateNames, AllStateData, AllEventData,
                  AllEventTimestamps, AllTrialStarts, AllTrialStarts, AllRawEvents))))

    # AllStimTrials optional special case
    if 'StimTrials' in SessionData._fieldnames:
        log.debug('StimTrials detected in session - will include')
        AllStimTrials = SessionData.StimTrials
        assert (AllStimTrials.shape[0] == AllStateTimestamps.shape[0])
    else:
        log.debug('StimTrials not detected in session - will skip')
        AllStimTrials = np.array([
            None for _ in enumerate(range(AllStateTimestamps.shape[0]))])

    # AllFreeTrials optional special case
    if 'FreeTrials' in SessionData._fieldnames:
        log.debug('FreeTrials detected in session - will include')
        AllFreeTrials = SessionData.FreeTrials
        assert (AllFreeTrials.shape[0] == AllStateTimestamps.shape[0])
    else:
        log.debug('FreeTrials not detected in session - synthesizing')
        AllFreeTrials = np.zeros(AllStateTimestamps.shape[0], dtype=np.uint8)

    # Photostim Period: early-delay, late-delay (default is early-delay)
    # Infer from filename for now, only applicable to Susu's sessions (i.e. "SC" in h2o)
    # If RecordingRig3, then 'late-delay'
    photostim_period = 'early-delay'
    rig_name = re.search('Recording(Rig\d)_', matlab_filepath.name)
    if re.match('SC', h2o) and rig_name:
        rig_name = rig_name.groups()[0]
        if rig_name == "Rig3":
            photostim_period = 'late-delay'
    log.info('Photostim Period: {}'.format(photostim_period))

    trials = list(zip(AllTrialTypes, AllStimTrials, AllFreeTrials,
                      AllTrialSettings, AllStateTimestamps, AllStateNames,
                      AllStateData, AllEventData, AllEventTimestamps,
                      AllTrialStarts, AllRawEvents))

    if not trials:
        log.warning('skipping date {d}, no valid files'.format(d=date))
        return

    #
    # Trial data seems valid; synthesize session id & add session record
    # XXX: note - later breaks can result in Sessions without valid trials
    #

    assert skey['session_date'] == session_datetime.date()

    skey['session_date'] = session_datetime.date()
    skey['session_time'] = session_datetime.time()

    #
    # Actually load the per-trial data
    #
    log.info('BehaviorIngest.make(): trial parsing phase')

    # lists of various records for batch-insert
    rows = {k: list() for k in ('trial', 'behavior_trial', 'trial_note',
                                'trial_event', 'corrected_trial_event',
                                'action_event', 'photostim',
                                'photostim_location', 'photostim_trial',
                                'photostim_trial_event')}

    trial = namedtuple(  # simple structure to track per-trial vars
        'trial', ('ttype', 'stim', 'free', 'settings', 'state_times',
                  'state_names', 'state_data', 'event_data',
                  'event_times', 'trial_start', 'trial_raw_events'))

    trial_number = 0  # trial numbering starts at 1
    for t in trials:

        #
        # Misc
        #

        t = trial(*t)  # convert list of items to a 'trial' structure
        trial_number += 1  # increment trial counter
        log.debug('BehaviorIngest.make(): parsing trial {i}'.format(i=trial_number))

        # covert state data names into a lookup dictionary
        #
        # names (seem to be? are?):
        #
        # Trigtrialstart, PreSamplePeriod, SamplePeriod, DelayPeriod
        # EarlyLickDelay, EarlyLickSample, ResponseCue, GiveLeftDrop
        # GiveRightDrop, GiveLeftDropShort, GiveRightDropShort
        # AnswerPeriod, Reward, RewardConsumption, NoResponse
        # TimeOut, StopLicking, StopLickingReturn, TrialEnd
        #

        states = {k: (v + 1) for v, k in enumerate(t.state_names)}
        required_states = ('PreSamplePeriod', 'SamplePeriod',
                           'DelayPeriod', 'ResponseCue', 'StopLicking',
                           'TrialEnd')

        missing = list(k for k in required_states if k not in states)

        if len(missing):
            log.warning('skipping trial {i}; missing {m}'
                        .format(i=trial_number, m=missing))
            continue

        gui = t.settings.GUI

        # ProtocolType - only ingest protocol >= 3
        #
        # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
        # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
        #

        if 'ProtocolType' not in gui._fieldnames:
            log.warning('skipping trial {i}; protocol undefined'
                        .format(i=trial_number))
            continue

        protocol_type = gui.ProtocolType
        if gui.ProtocolType < 3:
            log.warning('skipping trial {i}; protocol {n} < 3'
                        .format(i=trial_number, n=gui.ProtocolType))
            continue

        #
        # Top-level 'Trial' record
        #

        tkey = dict(skey)
        startindex = np.where(t.state_data == states['PreSamplePeriod'])[0]
        endindex = np.where(t.state_data == states['TrialEnd'])[0]

        log.debug('states\n' + str(states))
        log.debug('state_data\n' + str(t.state_data))
        log.debug('startindex\n' + str(startindex))
        log.debug('endindex\n' + str(endindex))

        if not (len(startindex) and len(endindex)):
            log.warning('skipping {}: start/end mismatch: {}/{}'.format(
                trial_number, str(startindex), str(endindex)))
            continue

        try:
            tkey['trial'] = trial_number
            tkey['trial_uid'] = trial_number
            tkey['start_time'] = t.trial_start
            tkey['stop_time'] = t.trial_start + t.state_times[endindex][0]
        except IndexError:
            log.warning('skipping {}: IndexError: {}/{} -> {}'.format(
                trial_number, str(startindex), str(endindex), str(t.state_times)))
            continue

        log.debug('tkey' + str(tkey))
        rows['trial'].append(tkey)

        #
        # Specific BehaviorTrial information for this trial
        #

        bkey = dict(tkey)
        bkey['task'] = 'audio delay'  # hard-coded here
        bkey['task_protocol'] = 1  # hard-coded here

        # determine trial instruction
        trial_instruction = 'left'  # hard-coded here

        if gui.Reversal == 1:
            if t.ttype == 1:
                trial_instruction = 'left'
            elif t.ttype == 0:
                trial_instruction = 'right'
        elif gui.Reversal == 2:
            if t.ttype == 1:
                trial_instruction = 'right'
            elif t.ttype == 0:
                trial_instruction = 'left'

        bkey['trial_instruction'] = trial_instruction

        # determine early lick
        early_lick = 'no early'

        if (protocol_type >= 5
                and 'EarlyLickDelay' in states
                and np.any(t.state_data == states['EarlyLickDelay'])):
            early_lick = 'early'
        if (protocol_type >= 5
                and ('EarlyLickSample' in states
                     and np.any(t.state_data == states['EarlyLickSample']))):
            early_lick = 'early'

        bkey['early_lick'] = early_lick

        # determine outcome
        outcome = 'ignore'

        if ('Reward' in states
                and np.any(t.state_data == states['Reward'])):
            outcome = 'hit'
        elif ('TimeOut' in states
              and np.any(t.state_data == states['TimeOut'])):
            outcome = 'miss'
        elif ('NoResponse' in states
              and np.any(t.state_data == states['NoResponse'])):
            outcome = 'ignore'

        bkey['outcome'] = outcome

        # Determine free/autowater (Autowater 1 == enabled, 2 == disabled)
        bkey['auto_water'] = int(gui.Autowater == 1 or np.any(t.settings.GaveFreeReward[:2]))
        bkey['free_water'] = t.free

        rows['behavior_trial'].append(bkey)

        #
        # Add 'protocol' note
        #
        nkey = dict(tkey)
        nkey['trial_note_type'] = 'protocol #'
        nkey['trial_note'] = str(protocol_type)
        rows['trial_note'].append(nkey)

        #
        # Add 'autolearn' note
        #
        nkey = dict(tkey)
        nkey['trial_note_type'] = 'autolearn'
        nkey['trial_note'] = str(gui.Autolearn)
        rows['trial_note'].append(nkey)

        #
        # Add 'bitcode' note
        #
        if 'randomID' in gui._fieldnames:
            nkey = dict(tkey)
            nkey['trial_note_type'] = 'bitcode'
            nkey['trial_note'] = str(gui.randomID)
            rows['trial_note'].append(nkey)

        # ==== TrialEvents ====
        trial_event_types = [('PreSamplePeriod', 'presample'),
                             ('SamplePeriod', 'sample'),
                             ('DelayPeriod', 'delay'),
                             ('ResponseCue', 'go'),
                             ('TrialEnd', 'trialend')]

        for tr_state, trial_event_type in trial_event_types:
            tr_events = getattr(t.trial_raw_events.States, tr_state)
            tr_events = np.array([tr_events]) if tr_events.ndim < 2 else tr_events
            for (s_start, s_end) in tr_events:
                ekey = dict(tkey)
                ekey['trial_event_id'] = len(rows['trial_event'])
                ekey['trial_event_type'] = trial_event_type
                ekey['trial_event_time'] = s_start
                ekey['duration'] = s_end - s_start
                rows['trial_event'].append(ekey)

                if trial_event_type == 'delay':
                    this_trial_delay_duration = s_end - s_start

        # ==== ActionEvents ====

        #
        # Add lick events
        #

        lickleft = np.where(t.event_data == 69)[0]
        log.debug('... lickleft: {r}'.format(r=str(lickleft)))

        action_event_count = len(rows['action_event'])
        if len(lickleft):
            [rows['action_event'].append(
                dict(tkey, action_event_id=action_event_count + idx,
                     action_event_type='left lick',
                     action_event_time=t.event_times[l]))
                for idx, l in enumerate(lickleft)]

        lickright = np.where(t.event_data == 71)[0]
        log.debug('... lickright: {r}'.format(r=str(lickright)))

        action_event_count = len(rows['action_event'])
        if len(lickright):
            [rows['action_event'].append(
                dict(tkey, action_event_id=action_event_count + idx,
                     action_event_type='right lick',
                     action_event_time=t.event_times[r]))
                for idx, r in enumerate(lickright)]

        # ==== PhotostimEvents ====

        #
        # Photostim Events
        #

        if photostim_period == 'early-delay':
            valid_protocol = protocol_type == 5
        elif photostim_period == 'late-delay':
            valid_protocol = protocol_type > 4

        if (t.stim and valid_protocol and gui.Autolearn == 4
                and np.round(this_trial_delay_duration, 2) == 1.2):
            log.debug('BehaviorIngest.make(): t.stim == {}'.format(t.stim))
            rows['photostim_trial'].append(tkey)
            if photostim_period == 'early-delay':  # same as the delay-onset
                delay_periods = t.trial_raw_events.States.DelayPeriod
                delay_periods = np.array(
                    [delay_periods]) if delay_periods.ndim < 2 else delay_periods
                stim_onset = delay_periods[-1][0]
            elif photostim_period == 'late-delay':  # 0.5 sec prior to the go-cue
                stim_onset = t.trial_raw_events.States.ResponseCue[0] - 0.5

            rows['photostim_trial_event'].append(
                dict(tkey,
                     photo_stim=t.stim,
                     photostim_event_id=len(
                         rows['photostim_trial_event']),
                     photostim_event_time=stim_onset,
                     power=5.5))

        # end of trial loop.

    return rows


def load_multi_target_licking_matfile(skey, matlab_filepath):
    """
    Loading routine for delay-response task - from .mat behavior data
    :param skey: session_key
    :param matlab_filepath: full-path to the .mat file containing the delay-response behavior data
    :return: nested list of all rows to be inserted into the various experiment-related tables
    """
    matlab_filepath = pathlib.Path(matlab_filepath)
    h2o = skey.pop('h2o')

    SessionData = spio.loadmat(matlab_filepath.as_posix(),
                               squeeze_me=True, struct_as_record=False)['SessionData']

    # parse session datetime
    session_datetime_str = str('').join((str(SessionData.Info.SessionDate), ' ',
                                         str(SessionData.Info.SessionStartTime_UTC)))
    session_datetime = datetime.strptime(
        session_datetime_str, '%d-%b-%Y %H:%M:%S')

    valid_trial_ind = np.where(SessionData.MATLABStartTimes != 0)[0]
    valid_trial_mapper = {trial_number: orig_trial_number
                          for trial_number, orig_trial_number
                          in zip(np.arange(len(valid_trial_ind)) + 1, valid_trial_ind + 1)} # "+ 1" as trial numbering starts at 1

    AllTrialTypes = SessionData.TrialTypes
    AllTrialSettings = SessionData.TrialSettings
    AllTrialStarts = SessionData.TrialStartTimestamp
    AllTrialStarts = AllTrialStarts - AllTrialStarts[0]  # real 1st trial

    RawData = SessionData.RawData
    AllStateNames = RawData.OriginalStateNamesByNumber
    AllStateData = RawData.OriginalStateData
    AllEventData = RawData.OriginalEventData
    AllStateTimestamps = RawData.OriginalStateTimestamps
    AllEventTimestamps = RawData.OriginalEventTimestamps

    AllRawEvents = SessionData.RawEvents.Trial

    # assign SessionBlock and SessionTrial association
    AllBlockTrials = np.cumsum(np.where(SessionData.TrialBlockOrder == 1, 1, 0))

    # index to only valid trials
    AllTrialTypes = AllTrialTypes[valid_trial_ind]
    AllTrialSettings = AllTrialSettings[valid_trial_ind]
    AllBlockTrials = AllBlockTrials[valid_trial_ind]

    # verify trial-related data arrays are all same length
    assert (all((x.shape[0] == AllStateTimestamps.shape[0] for x in
                 (AllTrialTypes, AllTrialSettings,
                  AllStateNames, AllStateData, AllEventData,
                  AllEventTimestamps, AllTrialStarts,
                  AllTrialStarts, AllRawEvents,
                  AllBlockTrials))))

    # AllStimTrials optional special case
    if 'StimTrials' in SessionData._fieldnames and len(SessionData.StimTrials):
        log.debug('StimTrials detected in session - will include')
        AllStimTrials = SessionData.StimTrials
        assert (AllStimTrials.shape[0] == AllStateTimestamps.shape[0])
    else:
        log.debug('StimTrials not detected in session - will skip')
        AllStimTrials = np.array([
            None for _ in enumerate(range(AllStateTimestamps.shape[0]))])

    # AllFreeTrials optional special case
    if 'FreeTrials' in SessionData._fieldnames:
        log.debug('FreeTrials detected in session - will include')
        AllFreeTrials = SessionData.FreeTrials
        assert (AllFreeTrials.shape[0] == AllStateTimestamps.shape[0])
    else:
        log.debug('FreeTrials not detected in session - synthesizing')
        AllFreeTrials = np.zeros(AllStateTimestamps.shape[0], dtype=np.uint8)

    trials = list(zip(AllTrialTypes, AllStimTrials, AllFreeTrials,
                      AllTrialSettings, AllStateTimestamps, AllStateNames,
                      AllStateData, AllEventData, AllEventTimestamps,
                      AllTrialStarts, AllRawEvents,
                      AllBlockTrials))

    if not trials:
        log.warning('skipping date {d}, no valid files'.format(d=date))
        raise InvalidBehaviorTrialError('skipping date {d}, no valid files'.format(d=date))

    assert skey['session_date'] == session_datetime.date()

    skey['session_date'] = session_datetime.date()
    skey['session_time'] = session_datetime.time()

    #
    # Actually load the per-trial data
    #
    log.info('BehaviorIngest.make(): trial parsing phase')

    # lists of various records for batch-insert
    rows = {k: list() for k in (
        'trial', 'behavior_trial', 'trial_note',
        'trial_event', 'corrected_trial_event',
        'action_event', 'photostim',
        'photostim_location', 'photostim_trial',
        'photostim_trial_event',
        'session_block', 'session_block_waterport', 'session_block_trial')}

    # simple structure to track per-trial vars
    trial = namedtuple('trial',
                       ('ttype', 'stim', 'free', 'settings', 'state_times',
                        'state_names', 'state_data', 'event_data',
                        'event_times', 'trial_start', 'trial_raw_events', 'trial_block'))

    trial_number = 0  # trial numbering starts at 1
    for t in trials:

        #
        # Misc
        #

        t = trial(*t)  # convert list of items to a 'trial' structure
        trial_number += 1  # increment trial counter

        log.debug('BehaviorIngest.make(): parsing trial {i}'.format(i=trial_number))

        states = {k: (v + 1) for v, k in enumerate(t.state_names)}

        gui = t.settings.GUI
        protocol_type = gui.ProtocolType

        #
        # Top-level 'Trial' record
        #

        tkey = dict(skey)
        startindex = np.where(t.state_data == states['TrigTrialStart'])[0]
        endindex = np.where(t.state_data == states['TrialEnd'])[0]

        log.debug('states\n' + str(states))
        log.debug('state_data\n' + str(t.state_data))
        log.debug('startindex\n' + str(startindex))
        log.debug('endindex\n' + str(endindex))

        if not (len(startindex) and len(endindex)):
            log.warning('skipping {}: start/end mismatch: {}/{}'.format(
                trial_number, str(startindex), str(endindex)))
            continue

        try:
            tkey['trial'] = trial_number
            tkey['trial_uid'] = trial_number
            tkey['start_time'] = t.trial_start
            tkey['stop_time'] = t.trial_start + t.state_times[endindex][0]
        except IndexError:
            log.warning('skipping {}: IndexError: {}/{} -> {}'.format(
                trial_number, str(startindex), str(endindex), str(t.state_times)))
            continue

        log.debug('tkey' + str(tkey))
        rows['trial'].append(tkey)

        #
        # Specific Multi-target-licking blocks information for this trial
        #
        trial_block = t.trial_block
        block_key = {**skey, 'block': trial_block}

        # check if first trial in block
        if (not rows['session_block']
                or rows['session_block'][-1]['block'] != trial_block):

            auto_water = True  # first trial in a block is always auto-water trial
            block_trial_number = 1

            # block details
            rows['session_block'].append({
                **block_key,
                'block_start_time': tkey['start_time'],
                'trial_count': sum(AllBlockTrials == trial_block),
                'num_licks_for_reward': gui.NumLicksForReward,
                'roll_deg': gui.RollDeg,
                'position_x_bins': gui.num_bins,
                'position_y_bins': 1,
                'position_z_bins': gui.num_bins
            })
            # block's water-port details
            row_idx, col_idx = np.where(SessionData.trial_type_mat[trial_number-1] == t.ttype)

            rows['session_block_waterport'].append({
                **block_key,
                'water_port': 'mtl-{}'.format(t.ttype),
                'position_x': SessionData.X_positions_mat[trial_number-1][row_idx[0], col_idx[0]],
                'position_y': gui.Y_center,
                'position_z': SessionData.Z_positions_mat[trial_number-1][row_idx[0], col_idx[0]]
            })
        else:
            auto_water = False
            block_trial_number = rows['session_block_trial'][-1]['block_trial_number'] + 1

        # block's trial details
        rows['session_block_trial'].append({
            **block_key, **tkey,
            'block_trial_number': block_trial_number})

        #
        # Specific BehaviorTrial information for this trial
        #

        bkey = dict(tkey)
        bkey['task'] = 'multi-target-licking'  # hard-coded here
        bkey['task_protocol'] = int(protocol_type)

        # trial instruction - 'none' - no notion of "trial instruction" for this task
        bkey['trial_instruction'] = 'none'  # hard-coded here

        # determine early lick - no notion of "early lick" for this task
        bkey['early_lick'] = 'no early'

        # determine outcome - no notion of "outcome" for this task
        bkey['outcome'] = 'N/A'

        # Determine autowater (Autowater 1 == enabled, 2 == disabled)
        auto_water = gui.Autowater == 1 or auto_water
        bkey['auto_water'] = int(auto_water)

        # Determine freewater
        bkey['free_water'] = int(bool(t.free))

        rows['behavior_trial'].append(bkey)

        #
        # Add 'protocol' note
        #
        nkey = dict(tkey)
        nkey['trial_note_type'] = 'protocol #'
        nkey['trial_note'] = str(protocol_type)
        rows['trial_note'].append(nkey)

        #
        # Add 'bitcode' note
        #
        # binary representation of the trial-number as a string
        # using the original trial numbering (to match with ephys's bitcode)
        orig_trial_number = valid_trial_mapper[tkey['trial']]
        nkey = dict(tkey)
        nkey['trial_note_type'] = 'bitcode'
        nkey['trial_note'] = np.binary_repr(orig_trial_number, 10)[::-1]
        rows['trial_note'].append(nkey)

        # ==== TrialEvents ====
        trial_event_types = [('TrialEnd', 'trialend')]

        for tr_state, trial_event_type in trial_event_types:
            tr_events = getattr(t.trial_raw_events.States, tr_state)
            tr_events = np.array([tr_events]) if tr_events.ndim < 2 else tr_events
            for (s_start, s_end) in tr_events:
                ekey = dict(tkey)
                ekey['trial_event_id'] = len(rows['trial_event'])
                ekey['trial_event_type'] = trial_event_type
                ekey['trial_event_time'] = s_start
                ekey['duration'] = s_end - s_start
                rows['trial_event'].append(ekey)

        # ==== ActionEvents ====
        # no action events for multi-target-licking task

        # end of trial loop.

    return rows

