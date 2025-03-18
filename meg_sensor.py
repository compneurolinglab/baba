import os
import numpy as np
import mne
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from autoreject import Ransac
from copy import deepcopy
from mne.preprocessing import ICA
import pickle
from scipy.stats import zscore
import math
from pyprep.prep_pipeline import PrepPipeline
import matplotlib.pyplot as plt


raw = mne.io.read_raw_fif('/Volumes/T7_Shield/meg/被试30meg/30.fif', preload=True)
#raw.notch_filter(freqs=[50, 100, 150], fir_design='firwin')
raw.filter(1,38, method='iir')
raw.plot()

raw.plot_sensors()
raw.plot_sensors(sphere=(0,0,0,0.1))

bad_chan=[]
raw.info['bads'] = bad_chan
raw.interpolate_bads(reset_bads=True)

ica = mne.preprocessing.ICA(method="fastica",random_state=42,n_components=20)
ica.fit(raw)
ica.plot_sources(raw, show_scrollbars=True)
ica.plot_components(sphere=(0,0,0,0.1))

ica.exclude = [1,3,12]
raw = ica.apply(raw, exclude=ica.exclude)
raw.save('/Volumes/T7_Shield/meg/preproc/subj30_ICA-raw.fif',overwrite=True)

events = mne.find_events(raw, stim_channel='STI1')  
time1 = events[0,0]

last_two_rows = events[-2:]
time1 = events[1,0]

df_onsets = pd.read_csv('/Volumes/T7_Shield/baba/raw/scu/raw_fif/scu_bhv_onset.csv') # to ms
df_onsets = pd.read_csv('/Users/elaine/Desktop//scu_bhv_onset.csv')
q1_onset = time1 + int(df_onsets['q1_onset'][20]*1000)  # q1开始时间   注意[]要改动
q5_offset = int(time1 + df_onsets['q5_onset'][20]*1000 + df_onsets['q5_rt'][20]*1000)    # q5结束时间  注意[]要改动
new_row = np.array([q1_onset, 0, 6])
updated_events = np.insert(events, 1, new_row, axis=0)
time2 = updated_events[1,0]
time3 = updated_events[2,0]
np.save('/Volumes/T7_Shield/meg/preproc/subj19_events.npy',updated_events)


raw = mne.io.read_raw_fif('/Volumes/T7_Shield/meg/preproc/subj19_ICA-raw.fif',preload=True)
# get task
epoch_tmin1 = time1
epoch_tmax1 = time1 + int((25*60+19) * 1000)   # exact meg onset & video length
raw.copy().crop(epoch_tmin1/1000,epoch_tmax1/1000)
events = np.array([[epoch_tmin1/1000,0,1]]) # event time
events = events.astype(int)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
epochs = mne.Epochs(raw,events,tmin=epoch_tmin1/1000,tmax=epoch_tmax1/1000,picks=picks,decim=1,baseline=None,preload=True)
epochs.save('/Volumes/T7_Shield/meg/preproc/subj19_task_epo.fif',overwrite=True)

raw = mne.io.read_raw_fif('/Volumes/T7_Shield/meg/preproc/subj19_ICA-raw.fif',preload=True)
# get question
epoch_tmin2 = time2
epoch_tmax2 = q5_offset
raw.copy().crop(epoch_tmin2/1000,epoch_tmax2/1000)
events = np.array([[epoch_tmin2/1000,0,1]]) # event time
events = events.astype(int)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
epochs = mne.Epochs(raw,events,tmin=epoch_tmin2/1000,tmax=epoch_tmax2/1000,picks=picks,decim=1,baseline=None,preload=True)
epochs.save('/Volumes/T7_Shield/meg/preproc/subj19_question_epo.fif',overwrite=True)

raw = mne.io.read_raw_fif('/Volumes/T7_Shield/meg/preproc/subj19_ICA-raw.fif',preload=True)
# get reply
#epoch_tmax = raw.n_times/1000-100   # 15 mins + 100ms
epoch_tmin3 = time3    # second trigger
epoch_tmax3 = time3 + 15*60*1000
raw.copy().crop(epoch_tmin3/1000,epoch_tmax3/1000)
events = np.array([[epoch_tmin3/1000,0,1]]) # event time
events = events.astype(int)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
epochs = mne.Epochs(raw,events,tmin=epoch_tmin3/1000,tmax=epoch_tmax3/1000,picks=picks,decim=1,baseline=None,preload=True)
epochs.save('/Volumes/T7_Shield/meg/preproc/subj19_replay_epo.fif',overwrite=True)