import os
import numpy as np
import nibabel as nib
from nilearn import image, datasets, surface, plotting
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import resample_to_img
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DIR = '/scratch/ResearchGroups/lt_jixingli/baba'
os.chdir(DIR)

affine = np.load('fmri/derivatives/affine.npy')
header = np.load('fmri/derivatives/header.npy', allow_pickle=True)

n_subj = 60
isc, isc_imgs = [], []
for i in range(31, n_subj + 1):
    isc_subj = np.load(f'fmri/isc/sub-{i}_isc_mean.npy')
    isc_subj = np.nan_to_num(zscore(isc_subj, nan_policy='omit'))   
    isc_subj = isc_subj.reshape((97, 115, 97))
    isc_subj_img = nib.Nifti1Image(isc_subj, affine)
    isc.append(isc_subj)
    isc_imgs.append(isc_subj_img)

design_matrix = pd.DataFrame([1]*len(isc_imgs),columns=['intercept'])   
second_level_model = SecondLevelModel(smoothing_fwhm=8,n_jobs=4)
second_level_model.fit(isc_imgs,design_matrix=design_matrix)
zmap = second_level_model.compute_contrast(second_level_contrast='intercept',output_type='z_score')
zmap,_ = threshold_stats_img(zmap,height_control='fdr')
