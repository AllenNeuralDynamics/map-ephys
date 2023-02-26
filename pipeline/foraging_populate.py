import datajoint as dj
from datetime import datetime
import traceback
import time

import sys
sys.path.append('/root/capsule/code')

from pipeline import (lab, get_schema_name, foraging_analysis, report, psth_foraging, 
                      foraging_model, ephys, experiment, foraging_analysis_and_export)
from pipeline.export import to_s3

import multiprocessing as mp
from threading import Timer, Thread

# Ray does not support Windows, use multiprocessing instead
use_ray = False

# My tables
my_tables = [       
        # Round 0 - old behavioral tables
        [
            foraging_analysis.TrialStats,  # Very slow
            foraging_analysis.BlockStats,
            foraging_analysis.SessionTaskProtocol,  #  Important for model fitting
            foraging_analysis.SessionStats,
            foraging_analysis.BlockFraction,
            foraging_analysis.SessionMatching,
            foraging_analysis.BlockEfficiency,
            foraging_analysis.SessionEngagementControl,
            ],
        # Round 1 - model fitting
        [
            foraging_model.FittedSessionModel,
            foraging_model.FittedSessionModelComparison,
            psth_foraging.UnitPeriodActivity,
            ],
        # Round 2 - ephys
        [
            psth_foraging.UnitPeriodLinearFit,
            # ephys.UnitWaveformWidth,
        ],
        # Round 3 - reports
         [
            # report.SessionLevelForagingSummary,
            # report.SessionLevelForagingLickingPSTH
        #     report.UnitLevelForagingEphysReportAllInOne
            foraging_analysis_and_export.SessionLogisticRegression,
            foraging_analysis_and_export.SessionBehaviorFittedChoiceReport,
            foraging_analysis_and_export.SessionLickPSTHReport,
            foraging_analysis_and_export.SessionWSLSReport,
            foraging_analysis_and_export.SessionLinearRegressionRT,
         ]
        ]

def populatemytables_core(arguments, runround):
    dj.conn().connect()
    for table in my_tables[runround]:
        table.populate(**arguments)
        
def show_progress(rounds=range(len(my_tables))):
    print('\n--- Current progress ---', flush=True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    for runround in rounds:
        for table in my_tables[runround]:
            dj.conn().connect()
            # table.progress()
            finished_in_current_key_source = len(table.key_source.proj() & table)
            total_in_current_key_source = len(table.key_source.proj())
            print(f"{f'{table.__name__}: ':<30}"
                  f"{f'{finished_in_current_key_source} / {total_in_current_key_source} = {finished_in_current_key_source / total_in_current_key_source:.3%},':<40}"
                  f"{f' to do: {total_in_current_key_source - finished_in_current_key_source}':<10}, ",
                  flush=True, end='')
            
            try:
                last_session_date = max((experiment.Session & table).fetch('session_date'))
                print(f'last date = {last_session_date}', flush=True)
            except:
                pass

        print(f'', flush=True)
    print('------------------------\n', flush=True)
        
def populatemytables(pool = None, cores = 9, all_rounds = range(len(my_tables))):
    # show_progress(all_rounds)
    
    if pool is not None:
        # schema = dj.schema(get_schema_name('foraging_analysis'),locals())
        # schema.jobs.delete()
    
        arguments = {'display_progress' : False, 'reserve_jobs' : True, 'suppress_errors': True}
        for runround in all_rounds:
            print('--- Parallel round '+str(runround)+'---', flush=True)
            
            result_ids = [pool.apply_async(populatemytables_core, args = (arguments,runround)) for coreidx in range(cores)] 
            
            for result_id in result_ids:
                result_id.get()

            print('  round '+ str(runround)+'  done...', flush=True)
    
        # show_progress(all_rounds)
        
    # Just in case there're anything missing?          
    print('--- Run with single core...', flush=True)
    for runround in all_rounds:
        print('   round '+str(runround)+'', flush=True)
        arguments = {'display_progress' : True, 'reserve_jobs' : True, 'order': 'random'}
        populatemytables_core(arguments, runround)
        
    # Show progress
    # show_progress(all_rounds)

def export_to_s3():
    experiment.PhotostimForagingLocation.load_photostim_foraging_location()
    to_s3.export_df_foraging_sessions()
    to_s3.export_df_regressions()
    to_s3.export_df_model_fitting_param()

def clear_jobs():
    for schema in [foraging_model, foraging_analysis, psth_foraging, report, ephys]:
        s = schema.schema
        print(f'{schema}: {len(s.jobs)} remaining jobs cleaned!')
        s.jobs.delete()
    return
    
    
class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            try:
                self.function(*self.args, **self.kwargs)
            except Exception as e:
                print(traceback.format_exc())
            
def run_with_progress(cores=None, run_count=1, print_interval=60):
    
    if cores is None:
        cores = int(mp.cpu_count()) - 1  # Auto core number selection

    print(f'# workers = {cores}')

    if cores > 1:
        pool = mp.Pool(processes=cores)
    else:
        pool = None
    
    # t1 = threading.Thread(target=populatemytables, 
    #                       kwargs=dict(pool=pool, cores=cores, all_rounds=range(len(my_tables)))
    # ) 
    
    back_ground_tasks = [[export_to_s3, 3600 * 24],
                         [show_progress, print_interval],
                         [clear_jobs, 3600 * 24],
                         ]    # (function, interval in s)

    tasks = []
    for task, interval in back_ground_tasks:  
        task()          
        tasks.append(RepeatTimer(interval, task))

    [task.start() for task in tasks]

    while run_count:
        try:
            run_count -= 1
            populatemytables(pool=pool, cores=cores, all_rounds=range(len(my_tables)))
            time.sleep(60 * 10)
        except:
            pass
    
    [task.join() for task in tasks]
    
    if pool != '':
        pool.close()
        pool.join()
            
            
if __name__ == '__main__' and use_ray == False:  # This is a workaround for mp.apply_async to run in Windows

    # from pipeline import shell
    # shell.logsetup('INFO')
    # shell.ingest_foraging_behavior()
    
    run_with_progress(cores=None, run_count=-1, print_interval=60 * 10)
    
 