# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:44:39 2022

@author: kenta.hagihara
"""

import datajoint as dj
dj.config['database.host'] = 'datajoint.mesoscale-activity-map.org'
dj.config['database.user'] = 'kenta'
dj.config['database.password'] = 'simple'#ToBeChanged
dj.conn().connect()

#%%
dj.conn()

#%%
#from pipeline import ophys
from pipeline import (lab, experiment, ophys)



#%%
#(experiment.TrialNote()  & (lab.WaterRestriction & 'water_restriction_number = "KH_FB8"') & 'session=1' & 'trial_note_type = "bitcode"').fetch('trial_note')
#(experiment.TrialNote & key & 'trial_note_type = "bitcode"').fetch('trial_note')

#%%
#map_schemas = [s for s in dj.list_schemas() if 'map_v2' in s]
#dj.Diagram(map_schemas)

#%%
#s = dj.schema('map_v2_lab')
#s = dj.schema('map_v2_experiment')
#s = dj.schema('map_v2_ophys')
#dj.Diagram(s)