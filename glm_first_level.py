import os, sys
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

DIR = '/scratch/ResearchGroups/lt_jixingli/baba'
os.chdir(DIR)

subj_id = int(sys.argv[1])
subj = 'sub-0%s' %subj_id if subj_id < 10 else 'sub-%s' %subj_id
regs = pd.read_csv('Analysis/hrf_reg.csv',header=None)
X = np.nan_to_num(zscore(regs,axis=0,nan_policy='omit'))
fmri_img = nib.load('Data/derivatives/%s/func/%s_task-baba_desc-preproc_bold.nii.gz' %(subj,subj))
fmri_data = fmri_img.get_fdata().reshape(-1,760) 

betas = np.zeros((fmri_data.shape[0],4))
for i in range(fmri_data.shape[0]):
	y = np.nan_to_num(zscore(fmri_data[i,:],nan_policy='omit'))
	betas[i] = LinearRegression().fit(X,y).coef_
np.save('Results/glm/subj%s_beta.npy' %subj_id, betas)
