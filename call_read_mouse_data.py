import sys
import os
import glob
import pandas as pd

data_dir = "../../Data/Y_maze/"

data_target_dir = "../../Data/processed_data/Y_maze/" 


files = glob.glob(data_dir+"*.mat")
print(files)
for f in files:
	exp_type = f.split(".mat")[0].split('/')[-1]
	print("experiment_type",exp_type)
	os.system("python read_mouse_data_full.py "+exp_type)


files_process = glob.glob(data_target_dir+"*_processed.csv")
all_conditions = pd.DataFrame()
for f in files_process:
	if "all_conditions_processed.csv" in f:
		print("ignoring",f)
		continue
	temp = pd.read_csv(f)
	all_conditions = all_conditions.append(temp,ignore_index=True)

all_conditions.to_csv(data_target_dir+"all_conditions_processed.csv")

	
