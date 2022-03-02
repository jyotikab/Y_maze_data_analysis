import os
import pdb
from simulation_functions_loki import Simulation
import numpy as np
from time import time
import glob
import pandas as pd
from pickle_objects import save_object
import sys
import gc


def calculation_b_cpp(path):

    def cpu_simulation(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring,  condition_code,conflict_code,volatility_code,session):
        print("calling Simulation")
        sim = Simulation(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring,condition_code, conflict_code, volatility_code,session)
        print("calling calc_B_CPP()")
        sim.calc_B_CPP()
        print("calc_B_cpp done")

        return sim

    exp_parameter_path = path 

    sim_data_path =  exp_parameter_path+"/simulated_data/"
        
    if os.path.exists(exp_parameter_path) == False:
        os.mkdir(exp_parameter_path)
    if os.path.exists(sim_data_path) == False:
        os.mkdir(exp_parameter_path+"/"+"simulated_data")

    mod_alpha = 1.0 # From human data equivalent 
    mod_beta = 0.143

    learning_rates = {'beta_drift': 1.0,'beta_boundary': mod_beta}

    #condition_order_df = pd.read_csv(os.path.join(condition_key_path, 'reward_condition_order.csv')) # get condition sequence & skip subject that dropped the task
    subj_data_files = glob.glob(exp_parameter_path+'*[0-9].csv') # match pattern for exp. data file
    #print(subj_data_files)
    subj_data_files.sort()


            
    av_model = 3
    n_subjects = 4

    drift_start = 0.001
    bound_start = 0.3
    sp_start = 0.5
    tr_start = 0.2
    
    print("collecting args")
    args = [(av_model, learning_rates, drift_start,	bound_start, sp_start,tr_start, filen, filen.split('/')[-1].split('_')[0],filen.split('/')[-1].split('_')[1], filen.split('/')[-1].split('_')[2] , filen.split('/')[-1].split('_')[3].split('.')[0]  ) for filen in subj_data_files]

    sim_start_time = time()


    for i in np.arange(len(args)):
        print(i,"calling cpu_simulation")
        av_model_sims = cpu_simulation(*args[i])
        save_object(av_model_sims, sim_data_path + 'sim_condition_' + av_model_sims.condition_id + '_conflict_' + av_model_sims.conflict_id + '_volatility_' + av_model_sims.volatility_id +"_session_"+av_model_sims.session+'.pkl')
        del av_model_sims
        gc.collect()

    sim_end_time = time()


    sim_time = sim_end_time - sim_start_time



# save each simulation as a pickled object
#[save_object(sim, sim_data_path + 'sim' + sim.subject_id + '_reward' + sim.conflict_code + '_run' + sim.run_n + '.pkl') for sim in av_model_sims]
