import sys
import os
import numpy as np
import nibabel as nib
import pandas as pd
import mne, eelbrain
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_1samp_test
import mne
from mne.stats import combine_adjacency
from mne.channels.layout import _find_topomap_coords


DIR = '/Volumes/T7_Shield/SciData/'
Data_DIR = '/Volumes/T7_Shield/preproc/npy/'

n_subj = 30
all_data = []
for subj_id in range(1,n_subj+1):
    subj = 'sub-0%d' %subj_id if subj_id < 10 else 'sub-%d' %subj_id
    subj_data = np.load(Data_DIR + '%s_task-baba.npy' % (subj))  # 直接加载每个被试的数据
    all_data.append(subj_data)

all_data = np.array(all_data)  # (30,1,64,1519001) interval: 10ms
print(all_data.shape)
all_data = np.transpose(all_data, (0, 2, 1, 3))    # (30, 64, 1, 1519001)
print(all_data.shape)
data = all_data.reshape(30, 64, -1)  # (30,64,1519001)
print(data.shape)
mean_channel = np.median(data, axis=0)  # (64,1519001)
print(mean_channel.shape)
np.save(DIR + 'isc/meg_data.npy', data)
np.save(DIR + 'isc/meg_median_channel.npy', mean_channel)


n_channels = 64
isc_matrix = np.zeros((n_channels, n_subj))    #(64,30)
for c in range(n_channels):     # 遍历所有通道
    for i in range(n_subj):     # 遍历所有被试
        print('>>>>> preproc subj:%d/30' % (i + 1))
        subj_channel = data[i, c, :]    # 获取第 i 个被试的 c 通道时间序列
        isc_matrix[c, i], _ = pearsonr(subj_channel, mean_channel[c,:])  # 计算与 group 平均通道的皮尔逊相关性
print(isc_matrix.shape)
np.save(DIR + 'isc/meg_corr.npy', isc_matrix)

# corr = np.load('Results/isc/eeg_corr_%s.npy' %group)
corr = np.nan_to_num(zscore(isc_matrix, axis=0, nan_policy='omit'))
# raw = mne.io.read_raw_fif('Analysis/sample/sub-01_task-multitalker_eeg.vhdr', preload=True)
# info = raw.info
# montage = mne.channels.read_custom_montage('Analysis/CACS-64.bvef')  # 电极位置
# info.set_montage(montage)

fif_file_path = "/Volumes/T7_Shield/preproc/sub-20/sub-20_task-baba_desc-preproc_meg.fif"
epoch = mne.read_epochs(fif_file_path)
info = epoch.info
adjacency = combine_adjacency(n_channels)
print("Adjacency shape:", adjacency.shape)    #(64,64)
adjacency, ch_names = mne.channels.find_ch_adjacency(info, 'mag')   # 获取通道邻接矩阵

cor_exp = np.expand_dims(corr.T, axis=1)  # (30,1,64)
print(cor_exp)
t_val, clusters, p_val, H0 = spatio_temporal_cluster_1samp_test(cor_exp, n_permutations=10000,adjacency=adjacency, tail=1)
significant_channels = np.where(p_val < 0.05)[0]  # found 9 clusters
print("显著的通道索引:", significant_channels)   # [1，6]

# 结果可视化
# pos_2d = mne.viz.topomap._find_topomap_coords(info, picks=np.arange(len(info.ch_names)))
pos_2d = _find_topomap_coords(info, picks=np.arange(len(info.ch_names)))
fig, ax = plt.subplots()
im, cm = mne.viz.plot_topomap(t_val[0], pos_2d, axes=ax, outlines='head', vlim=(0, 5),sensors=True, show=False, cmap='Reds', contours=3)
ax_x_start = 0.9
ax_x_width = 0.02
ax_y_start = 0.05
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title('t-value', fontsize=10)
plt.subplots_adjust(left=0, bottom=0, right=0.9, top=1)
plt.savefig(DIR + 'plots/meg_cluster.png')
plt.show()


plt = epoch.plot_sensors(ch_type="mag")
plt.savefig(DIR + 'plots/meg_sensor.png')
plt.show()
