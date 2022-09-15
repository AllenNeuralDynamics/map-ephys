import datajoint as dj
from datetime import datetime
from pipeline import lab, get_schema_name, foraging_analysis, report, psth_foraging, foraging_model, ephys
import multiprocessing as mp

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
            ],
        # Round 1 - model fitting
        [
            foraging_model.FittedSessionModel,
            foraging_model.FittedSessionModelComparison
            ],
        # Round 2 - ephys
        [
            # psth_foraging.UnitPeriodLinearFit,
            ephys.UnitWaveformWidth,
        ],
        # Round 3 - reports
        [
            # report.SessionLevelForagingSummary,
            # report.SessionLevelForagingLickingPSTH
        ]
        ]

def populatemytables_core(arguments, runround):
    for table in my_tables[runround]:
        table.populate(**arguments)
        
def show_progress(rounds):
    print('\n--- Current progress ---', flush=True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    for runround in rounds:
        for table in my_tables[runround]:
            finished_in_current_key_source = len(table & table.key_source.proj())
            total_in_current_key_source = len(table.key_source.proj())
            print(f'{table.__name__}: '
                  f'{finished_in_current_key_source} / {total_in_current_key_source} = '
                  f'{finished_in_current_key_source / total_in_current_key_source:.3%},'
                  f'to do: {total_in_current_key_source - finished_in_current_key_source}',
                  flush=True)
    print('------------------------\n', flush=True)
        
def populatemytables(paralel = True, cores = 9, all_rounds = range(len(my_tables))):
    show_progress(all_rounds)
    
    if paralel:
        # schema = dj.schema(get_schema_name('foraging_analysis'),locals())
        # schema.jobs.delete()
    
        arguments = {'display_progress' : False, 'reserve_jobs' : True}
        for runround in all_rounds:
            print('--- Parallel round '+str(runround)+'---', flush=True)
            
            result_ids = [pool.apply_async(populatemytables_core, args = (arguments,runround)) for coreidx in range(cores)] 
            
            for result_id in result_ids:
                result_id.get()

            print('  round '+ str(runround)+'  done...', flush=True)
    
    show_progress(all_rounds)
        
    # Just in case there're anything missing?          
    print('--- Run with single core...', flush=True)
    for runround in all_rounds:
        print('   round '+str(runround)+'', flush=True)
        arguments = {'display_progress' : True, 'reserve_jobs' : False, 'order': 'random'}
        populatemytables_core(arguments, runround)
        
    # Show progress
    show_progress(all_rounds)
            
            
if __name__ == '__main__' and use_ray == False:  # This is a workaround for mp.apply_async to run in Windows

    # from pipeline import shell
    # shell.logsetup('INFO')
    # shell.ingest_foraging_behavior()
    
    cores = int(mp.cpu_count()) - 1  # Auto core number selection
    pool = mp.Pool(processes=cores)
    
    populatemytables(paralel=True, cores=cores, all_rounds=[0, 1])
    
    if pool != '':
        pool.close()
        pool.join()
