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

affine = np.load('/Volumes/T7_Shield/baba/derivatives/affine.npy')
header = np.load('/Volumes/T7_Shield/baba/derivatives/header.npy', allow_pickle=True)

n_subj = 30
isc, isc_imgs = [], []
for i in range(1, n_subj + 1):
    isc_subj = np.load('/Volumes/T7_Shield/SciData/fmri/isc/subj%d_isc_mean.npy' % i)
    isc_subj = np.nan_to_num(zscore(isc_subj, nan_policy='omit'))   
    isc_subj = isc_subj.reshape((97, 115, 97))
    isc_subj_img = nib.Nifti1Image(isc_subj, affine)
    isc.append(isc_subj)
    isc_imgs.append(isc_subj_img)

# second-level analysis
design_matrix = pd.DataFrame([1]*len(isc_imgs),columns=['intercept'])   
second_level_model = SecondLevelModel(smoothing_fwhm=8,n_jobs=4)
second_level_model.fit(isc_imgs,design_matrix=design_matrix)
zmap = second_level_model.compute_contrast(second_level_contrast='intercept',output_type='z_score')
zmap,_ = threshold_stats_img(zmap,height_control='fdr') 

# smooth image
zmap_data = zmap.get_fdata()
zmap_data[zmap_data<0] = 0
zmap = nib.Nifti1Image(zmap_data,affine)
zmap = image.smooth_img(zmap,4)

# resample
gmask = nib.load('/Volumes/T7_Shield/副本/gm_no_cerebellum_brainstem.nii.gz')
mask = resample_to_img(gmask,zmap,interpolation='nearest')
img = nib.Nifti1Image(zmap.get_fdata() * mask.get_fdata(),affine=affine)

cmap = LinearSegmentedColormap.from_list('isc',['white','#fbcbb8','#f58b6c','#ee4934','#b62025'])
display = plotting.plot_stat_map(img, display_mode='z', black_bg=False,cut_coords=[-10,0,10,20,30,40], cmap=cmap, dim=2, vmax=10,symmetric_cbar=True)
display.savefig('/Volumes/T7_Shield/SciData/isc_slice.png')
plotting.show()

fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
surf_l = surface.vol_to_surf(img,surf_mesh=fsaverage["pial_left"],inner_mesh=fsaverage["white_left"])
surf_r = surface.vol_to_surf(img,surf_mesh=fsaverage["pial_right"],inner_mesh=fsaverage["white_right"])

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, surf_l,hemi='left', cmap='Reds',colorbar=True, bg_map=fsaverage.sulc_left,vmax=12, vmin=-2, symmetric_cbar=False)
fig.savefig('/Volumes/T7_Shield/SciData/isc_sufl.png')
plotting.show()

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, surf_r, hemi='right',cmap=cmap,symmetric_cbar=False, vmax=12, vmin=-2,colorbar=True, bg_map=fsaverage.sulc_right)
fig.savefig('/Volumes/T7_Shield/SciData/isc_sufr.png')
plotting.show()

