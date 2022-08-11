import os
import pdb
from simulation_functions_loki import Simulation
import numpy as np
from time import time
import glob
import pandas as pd
from pickle_objects import save_object
import sys


data_type = sys.argv[1] # cbgt2
exp_parameter_path = sys.argv[2]


home_path = os.path.expanduser('~')
if data_type == "ymaze": 
	#exp_parameter_path = "./Data/10/for_b_cpp_calculation/"
	sim_data_path =  exp_parameter_path+"/simulated_data/"
	print(sim_data_path)    
	if os.path.exists(exp_parameter_path) == False:
		#os.mkdir("./Data/10/for_b_cpp_calculation")
		os.mkdir(exp_parameter_path)
	if os.path.exists(sim_data_path) == False:
		os.mkdir(exp_parameter_path+"/"+"simulated_data")
elif data_type == "ymaze_binom": 
	#exp_parameter_path = "./Data/10/for_b_cpp_calculation/"
	sim_data_path =  exp_parameter_path+"/simulated_data_binom/"
	print(sim_data_path)    
	if os.path.exists(exp_parameter_path) == False:
		#os.mkdir("./Data/10/for_b_cpp_calculation")
		os.mkdir(exp_parameter_path)
	if os.path.exists(sim_data_path) == False:
		os.mkdir(exp_parameter_path+"/"+"simulated_data_binom")

        
mod_alpha = 1.0 # From human data equivalent 
mod_beta = 0.143

learning_rates = {'beta_drift': 1.0,'beta_boundary': mod_beta}


if "ymaze" in data_type:
	#subj_data_files = glob.glob(exp_parameter_path+'*[!block]_*[0-9]*[!with_b_cpp].csv')
	subj_data_files_temp = glob.glob(exp_parameter_path+'*[!block]_*[0-9]*.csv')
	subj_data_files = [x   for x in subj_data_files_temp if "with_b_cpp" not in x ]
    
 #glob.glob(exp_parameter_path+'*_[0-9]*.csv') # match pattern for exp. data file
    
	print(subj_data_files)
subj_data_files.sort()




if "ymaze" in data_type:

	def cpu_simulation( pathstring,  condition_code,conflict_code,volatility,session ):
		sim = Simulation( pathstring, condition_code,conflict_code,volatility,session)
		#pdb.set_trace()
		sim.calc_B_CPP()
		return sim
	

if "ymaze" in data_type:
	args = [( filen, filen.split('/')[-1].split('_')[0], filen.split('/')[-1].split('_')[1] , filen.split('/')[-1].split('_')[2], filen.split('/')[-1].split('_')[3].split('.csv')[0] ) for filen in subj_data_files]

sim_start_time = time()

#with Pool() as p:

if  "ymaze" in data_type:

	for i in np.arange(len(args)):
		av_model_sims = cpu_simulation(*args[i])
		save_object(av_model_sims, sim_data_path + 'sim' + '_conflict_' + av_model_sims.conflict_id + '_condition_' + av_model_sims.condition_id + "_volatility_"+av_model_sims.volatility_id+"_session_"+av_model_sims.session+'.pkl')


sim_end_time = time()


sim_time = sim_end_time - sim_start_time



# save each simulation as a pickled object
#[save_object(sim, sim_data_path + 'sim' + sim.subject_id + '_reward' + sim.conflict_code + '_run' + sim.run_n + '.pkl') for sim in av_model_sims]


