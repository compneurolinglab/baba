import os
import numpy as np
import nibabel as nib
from nilearn import image, datasets, surface, plotting
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import resample_to_img
from scipy.stats import zscore, pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn.reporting import get_clusters_table

DIR = '/scratch/ResearchGroups/lt_jixingli/baba'
os.chdir(DIR)

affine = np.load('Analysis/affine.npy')
header = np.load('Analysis/header.npy',allow_pickle=True)
anat =  nib.load(utils+'MNI152_T1_1mm_brain.nii.gz')
gray_matter_mask = nib.load('gm_no_cerebellum_brainstem.nii.gz')
groups = {'f0':f0,'intensity':intensity,'word':word,'switch':switch}

n_subjs = 30
f0,intensity,word,switch = [],[],[],[]
for i in range(n_subjs):
	beta = np.load('Results/glm/subj%s_beta.npy' %(i+1))
 	f0.append(nib.Nifti1Image(beta[:,0].reshape(97,115,97),affine=affine))
 	intensity.append(nib.Nifti1Image(beta[:,1].reshape(97,115,97),affine=affine))
 	word.append(nib.Nifti1Image(beta[:,2].reshape(97,115,97),affine=affine))
 	switch.append(nib.Nifti1Image(beta[:,3].reshape(97,115,97),affine=affine))
 
group = 'switch'
data = groups.get(group)
design_matrix = pd.DataFrame([1]*len(data),columns=['intercept'])
second_level_model = SecondLevelModel(smoothing_fwhm=8,n_jobs=4)
second_level_model.fit(data,design_matrix=design_matrix)
zmap = second_level_model.compute_contrast(second_level_contrast='intercept',output_type='z_score')

thresh_map,threshold = threshold_stats_img(zmap,height_control='fdr',cluster_threshold=10)
thresh_map_resampled = resample_to_img(thresh_map,gray_matter_mask,interpolation='nearest')
threshmap = thresh_map_resampled.get_fdata()*gray_matter_mask.get_fdata()
threshmap = nib.Nifti1Image(threshmap,affine=gray_matter_mask.affine)

stat_map = cluster_level_inference(zmap,threshold=2.5)
stat_map_resampled = resample_to_img(stat_map,gray_matter_mask,interpolation='nearest')
x = stat_map_resampled.get_fdata()*gray_matter_mask.get_fdata()
x = nib.Nifti1Image(x,affine=gray_matter_mask.affine)
