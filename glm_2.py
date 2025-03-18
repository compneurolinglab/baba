import os
import numpy as np
import nibabel as nib
from nilearn import image, datasets, surface, plotting
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import resample_to_img
import pickle
from scipy.stats import zscore, pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn.glm.thresholding import fdr_threshold
from nilearn.image import threshold_img
from scipy.ndimage import label

# DIR = '/scratch/yikwang5/baba/'
# os.chdir(DIR)

affine = np.load('/Volumes/T7_Shield/baba/derivatives/affine.npy')
gray_matter_mask = nib.load('/Volumes/T7_Shield/副本/gm_no_cerebellum_brainstem.nii.gz')
# gray_matter_mask = datasets.load_mni152_gm_mask()
plotting.plot_roi(gray_matter_mask, title="Gray Matter Mask")
plotting.show()

beta = nib.load('/Volumes/T7_Shield/SciData/fmri/glm/subj1_%s_beta.nii' %group)

# 读取被试的 Beta Map（第一层级 GLM 结果）
group = 'f0'
n_subj = 30
betas = []
for i in range(1,n_subj+1):
	beta = nib.load('/Volumes/T7_Shield/SciData/fmri/glm/subj%d_%s_beta.nii' %(i,group))   #(97,115,97)
	beta_data = zscore(beta.get_fdata().flatten(),nan_policy='omit')
	beta_data = np.nan_to_num(beta_data)
	beta_data = beta_data.reshape(beta.shape)
	beta = nib.Nifti1Image(beta_data,affine=affine)
	betas.append(beta)

# 执行第二层级（Group-Level）GLM
design_matrix = pd.DataFrame([1]*len(betas),columns=['intercept'])
second_level_model = SecondLevelModel(smoothing_fwhm=8,n_jobs=4)
second_level_model.fit(betas,design_matrix=design_matrix)   # 拟合GLM 模型，在群体水平计算统计参数。

# 计算 zmap（Z 统计值），衡量整体的激活显著性
zmap = second_level_model.compute_contrast(second_level_contrast='intercept',output_type='stat')
zmap_data = zmap.get_fdata()
zmap_data = -zmap_data
zmap_data[zmap_data<0]=0
zmap = nib.Nifti1Image(zmap_data,beta.affine)
plotting.plot_stat_map(zmap, title="Raw z-map")
plotting.show()

# # 进行簇级统计（Cluster-Level Inference）
stat_map = cluster_level_inference(zmap,threshold = 2.3,alpha=0.05)
stat_map_resampled = resample_to_img(stat_map,gray_matter_mask,interpolation='nearest')
x = stat_map_resampled.get_fdata() * gray_matter_mask.get_fdata()
x = nib.Nifti1Image(x,affine=gray_matter_mask.affine)
plotting.plot_stat_map(x)
plotting.show()

# stat_map, threshold = threshold_stats_img(zmap,threshold=3, alpha=0.05, height_control='fdr')
# stat_map_resampled = resample_to_img(stat_map,gray_matter_mask,interpolation='nearest')
# x = stat_map_resampled.get_fdata() * gray_matter_mask.get_fdata()
# x = nib.Nifti1Image(x,affine=gray_matter_mask.affine)
# plotting.plot_stat_map(x)
# plotting.show()

clists={'f0':'#f4bd0b','intensity':'#68ae67','switch':'#7d4f8c'}
cmap = LinearSegmentedColormap.from_list('%s'%group,['white',clists['%s'%group]])
fig = plt.figure()
display = plotting.plot_stat_map(zmap,display_mode='z',black_bg=False,vmax = 1, cmap=cmap,cut_coords=[-10,0,10,20,30,40],symmetric_cbar=False)
display.savefig('/Volumes/T7_Shield/SciData/%s.png' %(group))
nib.save(x,'/Volumes/T7_Shield/SciData/cluster_%s.nii' %(group))
plotting.show()

# 用nilearn画3d立体脑图
from nilearn import plotting
group = 'f0'
img = nib.load('/Volumes/T7_Shield/SciData/cluster_%s.nii'%(group))
plotting.plot_glass_brain(
    img,
    title='plot_glass_brain with display_mode="lyrz"',
    display_mode="lyrz",
	colorbar=True
)
plotting.show()
