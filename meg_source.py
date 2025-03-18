import os
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import create_ecg_epochs,create_eog_epochs,read_ica
from mne.coreg import Coregistration
from mne.io import read_info, read_fiducials
from mne.bem import make_scalp_surfaces

# export FREESURFER_HOME=/Applications/freesurfer
# export export SUBJECTS_DIR=/Volumes/Westside/baba/SCU/test/MRI/ #T1 MRI结构像所在路径
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# recon-all -s sub-04 -i /Volumes/Westside/baba/SCU/test/subj4/sub-04_T1w.nii -all
# mne watershed_bem -s sub-04
# mne make_scalp_surfaces -s sub-04 --overwrite --force

# source coregistration
subj = 0
subjects_dir = '/Volumes/T7_Shield/MRI/'
mne.bem.make_scalp_surfaces(subject='sub-30',subjects_dir=subjects_dir,overwrite=True)

raw = mne.io.read_raw_fif('/Volumes/T7_Shield/meg/preproc/sub-3%d/subj3%d_ICA-raw.fif' %(subj,subj), preload=True)
fiducials = read_fiducials('/Volumes/T7_Shield/meg/preproc/sub-3%d/subj3%d_ICA-raw.fif' %(subj,subj))[0]
info = raw.info


mne.gui.coregistration(subject='sub-3%d' %subj,subjects_dir=subjects_dir)     # save trans from gui
trans = mne.read_trans('/Volumes/T7_Shield/MRI/sub-3%d/sub-3%d-trans.fif' %(subj,subj))
plot_kwargs = dict(
    subject='sub-3%d' %subj,
    subjects_dir=subjects_dir,
    surfaces='head-dense',
    dig=True,
    eeg=[],
    meg='sensors',
    show_axes=True,
    coord_frame='meg',
)
view_kwargs = dict(azimuth=45,elevation=90,distance=0.6,focalpoint=(0.0, 0.0, 0.0))
mne.viz.plot_alignment(info,trans,**plot_kwargs)

# inverse
trans = mne.read_trans('/Volumes/T7_Shield/MRI/sub-3%d/sub-3%d-trans.fif' %(subj,subj))
src = mne.setup_source_space(subject='sub-3%d'%subj,spacing='ico4',subjects_dir=subjects_dir,n_jobs=8)
src.save('/Volumes/T7_Shield/MRI/sub-3%d/bem/sub-3%d-ico-4-src.fif' %(subj,subj), overwrite=True)
conductivity = (0.3,)
model = mne.make_bem_model(subject='sub-3%d'%subj,ico=4,conductivity=conductivity,subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
mne.write_bem_solution('/Volumes/T7_Shield/MRI/sub-3%d/bem/subj3%d-inner_skull-bem-sol.fif' %(subj,subj),bem,overwrite=True)
fwd = mne.make_forward_solution(info=info,trans=trans,src=src,bem=bem,ignore_ref=True,n_jobs=8)
mne.write_forward_solution('/Volumes/T7_Shield/MRI/sub-3%d/sub-3%d-fwd.fif' %(subj,subj),fwd, overwrite=True)
cov = mne.compute_raw_covariance(raw)
inv = mne.minimum_norm.make_inverse_operator(info,fwd,cov,fixed=True) 

# gen stc
snr = 3
lambda2 = 1.0/3.0**snr
times = np.load('MEG/sub-0%d/sub-0%d_times.npy' %(subj,subj))

events = np.array([
    [times[0],0,1],  # video_start
    [times[1],0,2],  # video_end
    [times[2],0,3],  # question_start
    [times[3],0,4],  # question_end
    [times[4],0,5],  # replay_start
    [times[5],0,6],  # replay_end
])
event_id = {'video':1,'video_end':2,'question':3,'question_end':4,'replay':5,'replay_end':6}

task1 = raw.copy().crop(times[0]/1000,times[1]/1000)
task2 = raw.copy().crop(times[2]/1000,times[3]/1000)
task3 = raw.copy().crop(times[4]/1000,times[5]/1000)

epoch1 = mne.Epochs(task1,events=events,event_id={'video':1}, 
                    tmin=0,tmax=(times[1]-times[0])/1000, 
                    baseline=(0,0),preload=True)
epoch1.save('MEG/sub-0%d/sub-0%d_baba-epo.fif' %(subj,subj))

epoch2 = mne.Epochs(task2,events=events,event_id={'question':3}, 
                    tmin=0,tmax=(times[3]-times[2])/1000, 
                    baseline=(0,0),preload=True)
epoch2.save('MEG/sub-0%d/sub-0%d_question-epo.fif' %(subj,subj))

epoch3 = mne.Epochs(task3,events=events,event_id={'replay':5}, 
                    tmin=0,tmax=(times[5]-times[4])/1000, 
                    baseline=(0,0),preload=True)
epoch3.save('MEG/sub-0%d/sub-0%d_replay-epo.fif' %(subj,subj))

stc1 = mne.minimum_norm.apply_inverse_epochs(epoch1,inv,lambda2=lambda2,method='dSPM')
stc1_morph = mne.compute_source_morph(stc1[0],subjects_dir=subjects_dir,spacing=4).apply(stc1[0])
stc1.save('STC/sub-0%d/sub-0%d_task-baba' %(subj,subj))

stc2 = mne.minimum_norm.apply_inverse_epochs(epoch2,inv,lambda2=lambda2,method='dSPM')
stc2_morph = mne.compute_source_morph(stc2[0],subjects_dir=subjects_dir,spacing=4).apply(stc2[0])
stc2_morph.save('STC/sub-0%d/sub-0%d_task-question' %(subj,subj),overwrite=True)

stc3 = mne.minimum_norm.apply_inverse_epochs(epoch3,inv,lambda2=lambda2,method='dSPM')
stc3.save('STC/sub-0%d/sub-0%d_task-replay' %(subj,subj))
