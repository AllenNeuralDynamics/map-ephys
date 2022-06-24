import os
import logging
import pathlib
import json
from datetime import datetime
from tqdm import tqdm
import re
from itertools import repeat
import pandas as pd
import numpy as np
import datajoint as dj

from .. import get_schema_name
from .. import lab, experiment, ephys, ophys

from pipeline.ingest.utils.paths import get_ophys_sess_dirs


schema = dj.schema(get_schema_name('ingest_ophys'))

log = logging.getLogger(__name__)


@schema
class OphysIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class OphysDir(dj.Part):
        ''' Path to the ophys directories (could be more than one if the recording has been interrupted) '''
        
        definition = """
        -> master
        ophys_dir:          varchar(255)    # yymmdd
        """

    key_source = (experiment.Session & ['subject_id = 619199']) - ophys.FiberPhotometrySession

    def make(self, key):
        '''
        Ophys .make() function
        '''

        log.info('\n======================================================')
        log.info('OphysIngest().make(): key: {k}'.format(k=key))
           

        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & key).fetch1()
        h2o = sinfo['water_restriction_number']
        log.info(sinfo)

        ophys_dirs = get_ophys_sess_dirs(key)

        if ophys_dirs == []:
            log.info(f'  No ophys folders found. Skipped ........................')
            return
        
        for i, ophys_dir in enumerate(ophys_dirs):  
            # TODO: concatenate across folders when the recording was disrupted.
            # Need to cache raw and ni_time etc. and do insertion after looping over all folders!
            
            log.info(f'  Ophys folder found {i + 1}/{len(ophys_dirs)}: {ophys_dir}')
            
            # --- Ingest meta data
            # Load meta file
            meta_file = list(ophys_dir.glob('meta*'))[0]
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Ingest
            fiber_meta = []
            roi_meta = []
            wavelen_meta = []
            
            # TODO: ingesting ophys.FiberPhotometrySession.Fiber from a meta csv file would be better!
            # Here I just add some fake info in meta.json
            for wavelen in meta['excitation_wavelength']:
                wavelen_meta.append({**key, **wavelen})
            
            for fiber in meta['fiber']:
                fiber_meta.append({**key, **fiber})
            
            for header, (fiber_num, color) in meta['roi'].items():
                roi_meta.append({**key, 'fiber_number': fiber_num, 'header_name': header, 'color': color})
                        
            ophys.FiberPhotometrySession.insert1({**key, 'ni_sample_rate': meta['sample_rate']}, allow_direct_insert=True)
            ophys.FiberPhotometrySession.ExcitationWavelength.insert(wavelen_meta)
            ophys.FiberPhotometrySession.Fiber.insert(fiber_meta)
            ophys.FiberPhotometrySession.ROI.insert(roi_meta)    
            log.info(f'  Meta info inserted!')
                    
            # --- Align timestamp to ni-samples---
            # TODO: please check this part carefully!!
            log.info(f'  Align frame times to ni time...')
            timestamp_file = list(ophys_dir.glob('TimeStamp*'))[0]
            timestamp = pd.read_csv(timestamp_file, names=['bonsai_time', 'frame_count'])
            
            syncpulse_file = list(ophys_dir.glob('TTL_TS*'))[0]
            syncpulse = pd.read_csv(syncpulse_file, names=['bonsai_time'])
            syncpulse_interval_ni_sample = 1 * meta['sample_rate']   # This is hard-coded, assuming syncpulse is 1Hz!
            
            frame_time_ni_sample = []
            
            for ts in timestamp['bonsai_time']:
                nearest_idx = np.argmin(np.abs(ts - syncpulse['bonsai_time']))  # find the nearest sync pulse
                time_diff_bonsai = ts - syncpulse['bonsai_time'][nearest_idx]  # bonsai-time difference between this frame and the nearest sync pulse
                time_diff_ni_sample = np.round(time_diff_bonsai / (1000 / meta['sample_rate']))  # diff in ms / ni_step_size in ms
                
                # This is relative to the first sync pulse! (could be negative if the first frame is before the first sync pulse)
                frame_time_ni_sample.append(nearest_idx * syncpulse_interval_ni_sample + time_diff_ni_sample)
                
            frame_time_ni_ms = np.array(frame_time_ni_sample) * (1000 / meta['sample_rate'])  # ni_sample * ni_step_size in ms
            frame_time_ni_ms = frame_time_ni_ms - frame_time_ni_ms[0]  # So that the time starts from zero (the first frame)
            log.info(f'     Done!')
                        
            # --- Ingest raw data and frame time (in ms, aligned to ni) ---
            # Read raw
            raw_file = list(ophys_dir.glob('Raw*'))[0]
            raw_pd = pd.read_csv(raw_file)
            
            log.info(f'  Ingest raw data...')
            
            for wavelen in wavelen_meta:
                for roi in roi_meta:
                    header_this = roi['header_name']
                    ledstate_this = wavelen['led_state']
                    
                    # raw for this
                    raw_this = raw_pd[raw_pd['LedState'] == ledstate_this][header_this] 
                    # ni_time for this
                    ni_time_this = frame_time_ni_ms[raw_this.index]
                    
                    # do insertion
                    ophys.FiberPhotometrySession.RawTrace.insert1({**key, **roi, **wavelen,
                                                                   'raw': raw_this.to_numpy(),
                                                                   'ni_time': ni_time_this},
                                                                  ignore_extra_fields=True)
        
            log.info(f'     Done!')
           

            # TODO: --- Insert ophys.TrialEvent and ophys.ActionEvent ---
            # 1. Decode events and the barcode
            # 2. Align barcode to behavior
            # 3. Do insertion
            
            
            pass
        
            
