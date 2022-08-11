#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import sys
#sys.path.append('/home/jyotika/Utils/hddm/')

#import hddm

data_dir = "./Data/processed_data/Y_maze/data_with_animal_ids/"
data_target_dir1 = "./Data/processed_data/Y_maze/data_with_animal_ids/for_b_cpp_calculation/"
data_target_dir2 = "./Data/processed_data/Y_maze/data_with_animal_ids/for_ddm_models/"
figure_dir = "./Figures/Y_maze/"

plt.rcParams["figure.facecolor"] = "w"
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# In[ ]:


#final_b_cpp_df = pd.read_csv(data_target_dir1+"for_av_fits_hddm.csv")
final_b_cpp_df = pd.read_csv(data_target_dir1+"for_av_fits_hddm_test.csv")


# In[ ]:


final_b_cpp_df


# In[ ]:


final_b_cpp_df.columns


# In[ ]:



block_change_aligned_df = pd.DataFrame()
for grp in final_b_cpp_df.groupby(["conflict","volatility","animal_id","condition","session"]):
    group_type = [ str(x) for x in grp[0]]
    print(grp[0])
    dat_slice = grp[1].copy()
    dat_slice["block_change"] = (grp[1]["block_num"]!=grp[1]["block_num"].shift()).astype(int)
    ind_block_change = np.where(dat_slice["block_change"])[0]
    print(ind_block_change)
    for x in ind_block_change[1:]:
        if dat_slice.iloc[x-1]["optimal"] == "left" and dat_slice.iloc[x]["optimal"] == "right":
            bc_type = "left->right"
        elif dat_slice.iloc[x-1]["optimal"] == "right" and dat_slice.iloc[x]["optimal"] == "left":
            bc_type = "right->left"
        bn_change = str(dat_slice.iloc[x-1]["block_num"])+"->"+str(dat_slice.iloc[x]["block_num"])
        for i in np.arange(x-2,x+10):
            if i > np.max(ind_block_change):
                continue
            #print(i)
            sub_slice = dat_slice.iloc[i]
            sub_slice["trials_from_change_point"] = i-x
            sub_slice["block_change_type"] = bc_type
            sub_slice["block_num_change"] = bn_change
            block_change_aligned_df = block_change_aligned_df.append(sub_slice)
    
block_change_aligned_df = block_change_aligned_df.reset_index(drop=True)   
block_change_aligned_df.iloc[np.where(block_change_aligned_df.index.duplicated())]
#fn = "_".join(group_type)
block_change_aligned_df.to_csv(data_target_dir2+"block_change_aligned_df.csv")


# In[ ]:


group_type


# In[ ]:


"_".join(group_type)


# In[ ]:





# In[ ]:


block_change_aligned_df


# In[ ]:



    

