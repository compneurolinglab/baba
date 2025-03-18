import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import nilearn
from numpy import genfromtxt
from scipy.stats import zscore
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_gm_mask
from scipy.signal import resample
from nilearn.glm.first_level import make_first_level_design_matrix
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix
from scipy.interpolate import interp1d

DIR = '/scratch/yikwang5/baba/'
os.chdir(DIR)

subjects = ['sub-%02d' % i for i in range(1, 31)] 
n_subjs = len(subjects)

subj_id = int(sys.argv[1]) - 1
subj = subjects[subj_id]

affine_path = os.path.join('Data/derivatives/affine.npy')
affine = np.load(affine_path)

img = nib.load(f'Data/derivatives/{subj}/func/{subj}_task-baba_desc-preproc_bold.nii.gz')
img_data = np.nan_to_num(zscore(img.get_fdata(),nan_policy='omit'))   
fmri_img = nib.Nifti1Image(img_data,affine=affine)

# gmask = load_mni152_gm_mask()
gmask = nib.load('Data/gm_no_cerebellum_brainstem.nii.gz')
mask = resample_to_img(gmask,fmri_img,interpolation='nearest') 

regs = ['f0','intensity','switch']
X = np.genfromtxt('Data/reg.csv', delimiter=',') 
X = np.nan_to_num(zscore(X,nan_policy='omit'))

contrast = np.eye(X.shape[1])
design_matrix = pd.DataFrame(X)
first_level_model = FirstLevelModel(t_r=1,hrf_model='spm',mask_img=mask)  
first_level_model.fit(fmri_img,design_matrices=design_matrix)
for i, reg_name in enumerate(regs):
    beta_map = first_level_model.compute_contrast(contrast[i], output_type='z_score')
    nib.save(beta_map, f'Results/glm/subj{subj_id+1}_{reg_name}_beta.nii')
    

	
