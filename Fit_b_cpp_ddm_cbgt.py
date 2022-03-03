#!/usr/bin/env python
# coding: utf-8

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
import os
import pickle
import glob
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, GroupKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import scipy as sp
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import pandas as pd
import numpy as np
import pylab as pl
import scipy.io as sio
import seaborn as sns
import statsmodels.api as sm
import funcs as func
import matplotlib.pyplot as plt
import sys
import pdb
import glob
import pickle
import binary_ddm_rl_simulation_p as binary_ddm
import sys
sys.path.append('/home/jyotika/Utils/hddm/')

import hddm

data_dir = "./Data/processed_data/Y_maze/"
data_target_dir = "./Data/processed_data/Y_maze/for_b_cpp_calculation/"
figure_dir = "./Figures/Y_maze/"

plt.rcParams["figure.facecolor"] = "w"
import sys


#from my_sklearn_tools.pca_regressors import LogisticPCR

from simulation_functions_loki import Simulation


# In[7]:


all_conflicts = pd.read_csv(data_dir+"all_experiments_df_with_DTs.csv")

conflict_dict = dict({'No':(0.9,0.),'Low':(0.9,0.1),'High':(0.75,0.25)}) 
# In[13]:


#for grp in all_conflicts.groupby(["conflict","volatility","session"]):
#    print(grp[1])


# In[15]:


#np.where(grp[1]["block"]!=grp[1]["block"].shift())[0]


# In[17]:


#np.where(grp[1]["block"]=="left")[0]


# In[20]:


def conv_rew_df_t_epochs_fmt(rew,cond,conf,vol,session):
    # Trial is constant, and first action is always left
    # For the general case, save the t_epochs df during the simulation
    # t0 == left
    #r_df = rew.loc[rew["data_type"]=="reward_df"]
    #pdb.set_trace()
    n_trials = len(rew)
    block = list(rew["block"])
    chosen_action = list(rew["chosen_action"])
    #reward = np.zeros(n_trials)
    #ind_rew = np.where(rew["rewarded"]=="rewarded")[0]
    #reward[ind_rew] = np.random.normal(loc=1.0,scale=0.01,size=len(ind_rew))
    reward = np.random.normal(loc=1.0,scale=0.01,size=n_trials)

    percentage = conflict_dict[conf]
    n_opt_trials = int(percentage[0]*n_trials)
    n_subopt_trials = int(percentage[1]*n_trials)

    opt_reward_idx = np.where(rew["optimal_action"]==1.0)[0]
    subopt_reward_idx = np.where(rew["optimal_action"]==0.0)[0]


    t_epochs = pd.DataFrame(columns=["r_t0","r_t1","cp","epoch_number","reward_p_t0","session","conflict","volatility","condition","p_id_solution","action_history","chosen_action","trial_num","RT(ms)"])
    cp = np.zeros(n_trials)
    #print(n_trials)
    cp_idx = np.where(rew["block"]!=rew["block"].shift())
    cp[cp_idx] = 1
    
        
    r_t0 =  np.zeros(n_trials)
    r_t1 = np.zeros(n_trials)

    #ind_t0 = np.where(np.logical_and(rew["block"]=="left",rew["rewarded"]=="rewarded"))[0]
    ind_t0 = np.where(rew["block"]=="left")[0]
    ind_t00 = np.sort(np.hstack([ np.random.choice(ind_t0[cp_idx[0][i]:cp_idx[0][i+1]],int(len(ind_t0[cp_idx[0][i]:cp_idx[0][i+1]])*percentage[0]),replace=False)   for i,x in enumerate(cp_idx[0]) if i < len(cp_idx[0])-1 ]))
    #ind_t1 = np.where(np.logical_and(rew["block"]=="right",rew["rewarded"]=="rewarded"))[0]
    ind_t1 = np.where(rew["block"]=="right")[0]
    ind_t10 = np.sort(np.hstack([ np.random.choice(ind_t1[cp_idx[0][i]:cp_idx[0][i+1]],int(len(ind_t1[cp_idx[0][i]:cp_idx[0][i+1]])*percentage[0]),replace=False)   for i,x in enumerate(cp_idx[0]) if i < len(cp_idx[0])-1 ]))

    r_t0[ind_t00] = reward[ind_t00]
    r_t1[ind_t10] = reward[ind_t10]

    t_epochs["r_t0"] = r_t0
    t_epochs["r_t1"] = r_t1
    t_epochs["cp"] = cp
    t_epochs["epoch_number"] = list(rew["block_num"])
    
    rew_p_t0 = np.zeros(n_trials)
    rew_p_t0[ind_t0] = 1.0
    t_epochs["reward_p_t0"] = rew_p_t0
    t_epochs["chosen_action"] = chosen_action
    t_epochs["action_history"] = [ 0 if x == "left" else 1 for x in chosen_action]
    t_epochs["p_id_solution"] = list(block)
    t_epochs["optimal"] = list(block)
    t_epochs["conflict"] = conf
    t_epochs["condition"] = cond
    t_epochs["volatility"] = vol
    t_epochs["session"] = session
    t_epochs["RT(ms)"] = list(rew["RT(ms)"])
    t_epochs["trial_num"] = list(rew["trial_num"])
    print(t_epochs)
    return t_epochs
    #print(t_epochs)
    
    


# In[ ]:


# Separate the batches into individual trials and save them as seperate .csv
for grp in all_conflicts.groupby(["conflict","volatility","session"]):
#     if seed == '2807188':
#         continue
    rew = grp[1].copy()
    cond = "Control"
    conf = grp[0][0]
    vol = grp[0][1]
    sess = grp[0][2]
    rew["rewarded_code"] = [ 1 if x == "rewarded" else 0 for x in rew["rewarded"]]
    #fig,ax = pl.subplots(1,1,figsize=(10,6))
    #sns.lineplot(x="index",y="rewarded_code",hue="block",data=rew,ax=ax)
    #fig.savefig(figure_dir+"Reward_df_"+cond+"_"+conf+"_"+vol+"_"+str(sess)+".png")
   
    
    b_cpp_compatible_format = conv_rew_df_t_epochs_fmt(rew,cond,conf,vol,sess)
    b_cpp_compatible_format = b_cpp_compatible_format.rename(columns={'RT(ms)':'rt'})

    #fig,ax = pl.subplots(1,1,figsize=(10,6))
    #ax.plot(b_cpp_compatible_format["r_t0"],label="r_t0")
    #ax.plot(b_cpp_compatible_format["r_t1"],label="r_t1")
    #ax.legend()
    #fig.savefig(figure_dir+"Reward_generated_df_"+cond+"_"+conf+"_"+vol+"_"+str(sess)+".png")
 

    filename = cond+"_"+conf+"_"+vol+"_"+str(sess)+".csv"
    b_cpp_compatible_format.to_csv(data_target_dir+filename)


# In[32]:



#data_target_dir1 = "../cbgt2_plasticity/Data/var_lambda/10/for_b_cpp_calculation/"
data_target_dir1 = "./Data/processed_data/Y_maze/for_b_cpp_calculation/"
print(os.getcwd())
#cmd = "python binary_ddm_rl_simulation_p.py "
#cmd = cmd + data_target_dir1
#output = os.system(cmd)
binary_ddm.calculation_b_cpp(data_target_dir1)


# In[33]:




# In[34]:



# In[38]:


#b_cpp_path = '/home/bahuguna/Work/CBGT_CMU/cbgt2_plasticity/Data/competition/for_b_cpp_calculation/simulated_data/'
b_cpp_path = data_target_dir1+'/simulated_data/'

files = glob.glob(b_cpp_path+"*[0-9]*.pkl")
import pickle

final_b_cpp_df = pd.DataFrame()

for f in files:
    if "with" in f:
        continue
    print(f)
    cond = f.split('/')[-1].split('sim_condition_')[1].split('_')[0]
    conf = f.split('/')[-1].split('_conflict_')[1].split('_')[0]
    volatility = f.split('/')[-1].split('volatility_')[1].split('_')[0]
    session = f.split('/')[-1].split('session_')[1].split('.pkl')[0]
    
    filesrc = cond+"_"+conf+"_"+volatility+"_"+session+".csv"
    filedest = cond+"_"+conf+"_"+volatility+"_"+session+"_with_b_cpp.csv"
    temp = pd.read_csv(data_target_dir1+filesrc)
        
    sim = pickle.load(open(f,"rb"))
    ideal_B = sim.ideal_B
    cpp = sim.CPP
    b_t0 = sim.B[:,0]
    b_t1 = sim.B[:,1]
    MC = sim.MC
    
    temp["cpp"] = cpp
    temp["ideal_B"] = ideal_B
    temp["b_t0"] = b_t0 
    temp["b_t1"] = b_t1
    temp["MC"] = MC
    temp.to_csv(data_target_dir+filedest)
    final_b_cpp_df = final_b_cpp_df.append(temp)
    
    
    print(temp)
    
final_b_cpp_df.to_csv(data_target_dir1+"for_av_fits_hddm.csv")


# In[ ]:





# In[39]:


final_b_cpp_df


# In[14]:


sns.lineplot


# In[40]:


final_b_cpp_df.loc[final_b_cpp_df["conflict"]=="Low"]


# In[ ]:




