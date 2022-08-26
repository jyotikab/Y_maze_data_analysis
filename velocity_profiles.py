#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pylab as pl
import scipy.io as sio
import seaborn as sns
import statsmodels.api as sm
import funcs as func
import sys
import pdb
import scipy.signal as sp_sig
import glob
import more_itertools as mit
import matplotlib.pyplot as plt
import os
import gc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cwd = os.getcwd()

print(cwd)
data_dir = cwd+"/Data/Y_maze/data_with_animal_ids/"
data_target_dir = cwd+"/Data/processed_data/Y_maze/data_with_animal_ids/"
figure_dir = cwd+"/Figures/"


# In[2]:


#experiment_type = "NoConfLowVolDTs"

plt.rcParams["figure.facecolor"] = "w"


# In[3]:


col_names = ["time (ms)","event_marker","current_trial_num_in_block","block","low_rew_%","high_rew_%","low_trial_range(block)","high_trial_range(block)","block#","rewarded_trial_count","total_trial_count","sig","x_pos","y_pos","trip","chnc","current_hall","previous_hall","lick_rate_hall1","lick_rate_hall2","lick_rate_hall3","wtr","mouse_distance","TTL"]

non_action_em = [0,11,12,9,10]
left_action_em = [1,2,3,4]
right_action_em = [5,6,7,8]
rewarded_em = [1,3,5,7]
unrewarded_em = [2,4,6,8]


# In[4]:



def read_raw_data(mat,ids):
    # block, 0 = left, 1 = right
    #columns = ["time (ms)", "event_markers", "current_trial_num_in_block","block","reward_probability","conflict","volatility","block_num","rewarded","action_chosen","condition"]
    r,c = np.shape(mat[0])
    raw_df = pd.DataFrame(columns = list(np.arange(0,c,1))+["session"])
    ids1 = np.hstack(ids)[0]
    print(ids1)
    for i in np.arange(len(mat)):
        temp = pd.DataFrame(mat[i])
        temp["session"] = i
        temp["animal_id"] = ids1[i]
        raw_df = raw_df.append(temp)

    return raw_df


def filter_ts(data,fs,fc):
#     fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = sp_sig.butter(5, w, 'low')
    output = sp_sig.filtfilt(b, a, data)
    
    return output

def calc_decision_times(raw_df,exp_type):
    xy = np.zeros((len(raw_df),2))
    xy[:,0] = raw_df["x_pos"].values
    xy[:,1] = raw_df["y_pos"].values
    conf = exp_type.split('Conf')[0]
    cond = exp_type.split('Vol')[1]
    #cond = "Control" # change when stim data is here !!!!!!
    vol = exp_type.split('Vol')[0].split('Conf')[1]

    speed = np.sqrt(np.diff(xy[:,0])**2+np.diff(xy[:,1])**2)
    speed = np.insert(speed,0,0)
    smooth_speed = np.convolve(speed,np.ones(30,dtype=int),'same')/30.
    
    
    fs = 1./(np.mean(np.diff(raw_df["time (ms)"].values)))*1000
    
    speed_filt = filter_ts(smooth_speed,fs,30)
    acc = np.gradient(speed_filt)
#     acc = np.insert(acc,0,0)
    flags = np.zeros((len(acc)))
#     ind = sp_sig.find_peaks(-speed_filt)[0] #what you need
    ind = np.where(np.abs(acc)<0.02)[0]
    flags[ind] = 1
    raw_df["speed"] = smooth_speed
    raw_df["speed_filt"] = speed_filt
    raw_df["acceleration"] = acc
    raw_df["DT_markers"] = flags
    raw_df["conflict"] = conf
    raw_df["volatility"] = vol
    raw_df["condition"] = cond
    ind_nomid = np.where(np.logical_and(raw_df["DT_markers"].values==1,raw_df["current_hall"]!=0))[0]
    trial_change = np.where(raw_df["current_trial_num_in_block"]!=raw_df["current_trial_num_in_block"].shift(1))[0] #,raw_df["current_hall"]!=0))[0]
    
    final_flag = np.zeros((len(acc)))
    for i,tc in enumerate(trial_change[:-1]):
        ind_nomid_trial = ind_nomid[np.logical_and(ind_nomid>=tc,ind_nomid<trial_change[1+i])]
        potential_dts = [list(group) for group in mit.consecutive_groups(ind_nomid_trial)]
        if len(potential_dts) > 1:
            longest = np.argmax([len(x) for x in potential_dts])
            lowest_speed = np.argmin([ np.mean(speed_filt[x]) for x in potential_dts])
            if longest == lowest_speed:
                final_flag[potential_dts[longest]] = 1
            else:
                final_flag[potential_dts[lowest_speed]] = 1
        else:
            final_flag[potential_dts] = 1
    raw_df["DT_markers_new"] = final_flag
    
    to_ret_df = raw_df[["session","block#","current_trial_num_in_block","time (ms)","speed","speed_filt","animal_id","DT_markers_new","conflict","volatility","condition","block","event_marker"]].copy()

    del raw_df
    gc.collect()
    return to_ret_df


# In[ ]:


velocity_profiles = pd.DataFrame()
exp_files = glob.glob(data_dir+"*.mat")
print(exp_files)
for ef in exp_files:
    print(ef)
    exp_type = ef.split('/')[-1].split('.')[0].split('_')[0]
    print(exp_type)
    exp_mat = sio.loadmat(ef)
    raw_df = read_raw_data(exp_mat['overlap'][0],exp_mat['overlap'][2])
    # Replace the column numbers with the labels that Julia gave
    raw_df = raw_df.rename(columns={i:cn for i,cn in zip(np.arange(len(col_names)),col_names)})
    
    to_ret_df = calc_decision_times(raw_df.copy(),exp_type)
    to_ret_df.to_csv(data_target_dir+"velocity_profiles_"+exp_type+".csv")
#     velocity_profiles = velocity_profiles.append(to_ret_df)

# velocity_profiles = velocity_profiles.reset_index()
# velocity_profiles.to_csv(data_target_dir+"velocity_profiles.csv")
    

    


# In[ ]:


# fs


# # In[ ]:


# xy[:,0] = raw_df["x_pos"].values
# xy[:,1] = raw_df["y_pos"].values

# speed = np.sqrt(np.diff(xy[:,0])**2+np.diff(xy[:,1])**2)
# conf = exp_type.split('Conf')[0]
# cond = exp_type.split('Vol')[1]
# #cond = "Control" # change when stim data is here !!!!!!
# vol = exp_type.split('Vol')[0].split('Conf')[1]
# speed = np.insert(speed,0,0)

# smooth_speed = np.convolve(speed,np.ones(30,dtype=int),'same')/30.
# fs = 1./(np.mean(np.diff(raw_df["time (ms)"].values)))*1000

# # What frequency should we filter ? 

# speed_filt = filter_ts(smooth_speed,fs,35)

# acc = np.gradient(speed_filt)
# #     acc = np.insert(acc,0,0)
# flags = np.zeros((len(acc)))
# #     ind = sp_sig.find_peaks(-speed_filt)[0] #what you need
# ind = np.where(np.abs(acc)<0.02)[0]
# flags[ind] = 1
# raw_df["speed"] = smooth_speed
# raw_df["speed_filt"] = speed_filt
# raw_df["acceleration"] = acc
# raw_df["DT_markers"] = flags
# raw_df["conflict"] = conf
# raw_df["volatility"] = vol
# raw_df["condition"] = cond
# ind_nomid = np.where(np.logical_and(raw_df["DT_markers"].values==1,raw_df["current_hall"]!=0))[0]
# trial_change = np.where(raw_df["current_trial_num_in_block"]!=raw_df["current_trial_num_in_block"].shift(1))[0] #,raw_df["current_hall"]!=0))[0]

# final_flag = np.zeros((len(acc)))
# for i,tc in enumerate(trial_change[:-1]):
#     ind_nomid_trial = ind_nomid[np.logical_and(ind_nomid>=tc,ind_nomid<trial_change[1+i])]
#     potential_dts = [list(group) for group in mit.consecutive_groups(ind_nomid_trial)]
#     if len(potential_dts) > 1:
#         longest = np.argmax([len(x) for x in potential_dts])
#         lowest_speed = np.argmin([ np.mean(speed_filt[x]) for x in potential_dts])
#         if longest == lowest_speed:
#             final_flag[potential_dts[longest]] = 1
#         else:
#             final_flag[potential_dts[lowest_speed]] = 1
#     else:
#         final_flag[potential_dts] = 1
# raw_df["DT_markers_new"] = final_flag
# #     new_raw_df = pd.DataFrame()
# #     for grp in raw_df.groupby(["session","block#","current_trial_num_in_block","animal_id"]):
# #         ind_nomid = np.where(np.logical_and(grp[1]["DT_markers"].values==1,grp[1]["current_hall"]!=0))[0] # Ignore if the animal slows down in the middle of Y-maze
# #         potential_dts = [list(group) for group in mit.consecutive_groups(ind_nomid)]
# # #         print(potential_dts)
# #         if len(potential_dts) > 1:
# #             longest = np.argmax([len(x) for x in potential_dts])
# #             new_dt = np.zeros((len(grp[1])))
# #             new_dt[potential_dts[longest]] = 1
# #             grp[1]["DT_markers_new"] = new_dt
# #             new_raw_df = new_raw_df.append(grp[1][["speed","speed_filt","DT_markers_new","time (ms)","session","block#","current_trial_num_in_block"]])
#     #print(ind_nomid)
    
    
    
    
    
    


# #     velocity_profiles = velocity_profiles.append(raw_df)


# # In[ ]:





# # In[ ]:





# # In[ ]:


# # raw_df["DT_markers_new"] = final_flag


# # In[ ]:


# potential_dts = [list(group) for group in mit.consecutive_groups(ind_nomid_trial)]


# # In[ ]:


# potential_dts


# # In[ ]:


# np.unique(raw_df["session"])


# # In[ ]:


# np.unique(raw_df["block#"])


# # In[ ]:


# import scipy.signal as sp_sig


# # In[ ]:


# sub_dat = raw_df.loc[(raw_df["session"]==1)&(raw_df["block#"] == 2)]
# # trial_change = np.where(sub_dat["current_trial_num_in_block"]!=sub_dat["current_trial_num_in_block"].shift(1))[0] #,raw_df["current_hall"]!=0))[0]
# # ind_nomid = np.where(np.logical_and(sub_dat["DT_markers"].values==1,sub_dat["current_hall"]!=0))[0]    
# # final_flag = np.zeros((len(sub_dat)))
# # for i,tc in enumerate(trial_change[:-1]):
# #     ind_nomid_trial = ind_nomid[np.logical_and(ind_nomid>=tc,ind_nomid<trial_change[1+i])]
# #     potential_dts = [list(group) for group in mit.consecutive_groups(ind_nomid_trial)]
# #     if len(potential_dts) > 1:
# #         longest = np.argmax([len(x) for x in potential_dts])
# #         final_flag[potential_dts[longest]] = 1
# #     else:
# #         final_flag[potential_dts] = 1

# # sub_dat["DT_markers_new"] = final_flag


# # In[ ]:


# final_flag[potential_dts[longest]]


# # In[ ]:


# # sub_dat = raw_df.loc[(raw_df["session"]==np.min(raw_df["session"]))&(raw_df["block#"] == np.min(raw_df["block#"]))]
# # for grp in sub_dat.groupby("current_trial_num_in_block"):
# fig,ax = pl.subplots(1,1,figsize=(20,10))
# sns.lineplot(x="time (ms)",y="speed",data=sub_dat,ax=ax)
# sns.lineplot(x="time (ms)",y="speed_filt",data=sub_dat,ax=ax)

# # ind = np.where(np.logical_and(sub_dat["DT_markers_new"].values==1),sub_dat["current_hall"]!=0))[0]
# ind = np.where(sub_dat["DT_markers_new"].values==1)[0] #,sub_dat["current_hall"]!=0))[0]
# # ind = sp_sig.argrelmin(sub_dat["speed"].values)
# ax.plot(sub_dat.iloc[ind]["time (ms)"],sub_dat.iloc[ind]["speed_filt"],'.',color='green')

# trial_change = np.where(np.logical_and(sub_dat["current_trial_num_in_block"]!=sub_dat["current_trial_num_in_block"].shift(1),sub_dat["current_hall"]!=0))[0]
# ax.plot(sub_dat.iloc[trial_change]["time (ms)"],sub_dat.iloc[trial_change]["speed_filt"],'*',color='firebrick',ms=10)
# # fig1,ax1 = pl.subplots(1,1,figsize=(20,10))

# # sns.lineplot(x="time (ms)",y="acceleration",data=sub_dat,ax=ax1)
# # ax1.plot(sub_dat.iloc[ind]["time (ms)"],sub_dat.iloc[ind]["acceleration"],'.',color='darkorange')
# # sns.lineplot(x="time (ms)",y="acceleration",data=sub_dat.loc[sub_dat["DT_markers"]==1],ax=ax1)


# # In[ ]:


# data = sub_dat["speed"].copy()
# fftfreq = np.fft.fftfreq(len(data),d=1./fs)
# fft = np.abs(np.fft.fft(data-np.mean(data)))

# pl.figure(figsize=(10,10))
# pl.plot(data)



# # In[ ]:


# pl.figure(figsize=(10,10))
# pl.plot(fftfreq[:int(len(fftfreq)/2)],fft[:int(len(fftfreq)/2)],'b-')

# pl.xlim(0,100)


# # In[ ]:


# np.argmax(fft[:int(len(fftfreq)/2)])


# # In[ ]:


# fftfreq[:int(len(fftfreq)/2)][15]


# # In[ ]:




