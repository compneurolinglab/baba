import sys
import os
import numpy as np
import nibabel as nib
import pandas as pd
import mne, eelbrain
import matplotlib as mpl
from scipy.stats import zscore, pearsonr, sem
from mne.stats import combine_adjacency, spatio_temporal_cluster_1samp_test
import matplotlib.pyplot as plt
from mne.channels import find_ch_adjacency
from mne.channels.layout import _find_topomap_coords

word = pd.read_csv('/Users/elaine/Desktop/Analysis/praat/baba.csv',usecols=[0,1,2,3,4])
n_subjs = 30

def get_word(subj_id):
	subj = f'sub-0{subj_id}' if subj_id < 10 else f'sub-{subj_id}'
	subj_data = np.load('/Volumes/T7_Shield/preproc/npy/%s_task-baba.npy'% (subj))[0,:,::10]  # (64,1519001)
	
	subj_word = []
	for idx, row in word.iterrows():
		onset = int(row['onset']*100)
		if onset+61 < subj_data.shape[1]:
			subj_word.append(subj_data[:,onset-10:onset+61])
		else:
			continue
	return subj_word 
all_data = []   
for i in range(1,n_subjs+1):
	print('>>> load data subj:%s' %i)
	subj_word = get_word(i)
	all_data.append(subj_word)

all_data = np.array(all_data) #(30, 3129, 64, 71)
np.save('/Users/elaine/Desktop/Results/evoked_all_word.npy', all_data)

all_data = np.load('/Users/elaine/Desktop/Results/evoked_all_word.npy')
mean_data = np.mean(all_data,axis=0) #((3129, 64, 71)
evk_data = np.mean(mean_data,axis=0) #(64,71)
subj_all = np.mean(all_data,axis=1) #(30,64,71)

zsubj_all = np.nan_to_num(zscore(subj_all,axis=1,nan_policy='omit'))
epochs = mne.read_epochs('/Volumes/T7_Shield/preproc/sub-20/sub-20_task-baba_desc-preproc_meg.fif',preload=True)
info = epochs.info
ch_names = [ch for ch in epochs.info["ch_names"] if "MEG" in ch]
adjacency = combine_adjacency(len(ch_names))

zsubj_all = zsubj_all.swapaxes(1,2) #(30,71,64)
t_val, clusters, p_val, H0 = spatio_temporal_cluster_1samp_test(zsubj_all, n_permutations=10000,adjacency=adjacency, tail=-1)
# 分别做tail=1和tail=-1

p_thrs = 0.9
sensors, times = [],[]
good_cluster_inds = np.where(p_val < p_thrs)[0]
if len(good_cluster_inds) > 0:
	for i_clu, clu_idx in enumerate(good_cluster_inds):
		time_inds, space_inds = np.squeeze(clusters[clu_idx])
		ch_inds = np.unique(space_inds)
		time_inds = np.unique(time_inds)
		sensors.append(ch_inds)
		times.append(time_inds)

sensor = sensors[0]
time = times[0]
tmap = t_val[time].mean(axis=0)    
tmap = np.nan_to_num(tmap)
mask = np.zeros(64)
mask[sensor]=1
tmask = np.where(mask==1,tmap,0)

# pos_2d = mne.viz.topomap._find_topomap_coords(info,picks=np.arange(len(info.ch_names)))
pos_2d = _find_topomap_coords(info, picks=np.arange(len(info.ch_names)))
fig,ax = plt.subplots()
mpl.rcParams['figure.figsize'] = [10,10]
im,cm = mne.viz.plot_topomap(tmask,pos_2d,axes=ax,outlines='head',vlim=(0,0.5),sensors=True,show=False,cmap='Blues',contours=3)
ax_x_start = 0.95
ax_x_width = 0.01
ax_y_start = 0.05
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
# clb.ax.set_title('t-value',fontsize=10)
plt.subplots_adjust(left=0.01, bottom=0, right=0.9, top=1)
plt.savefig('/Users/elaine/Desktop/Results/topo_evoke_2.png')
plt.show()

### plot evoke
linestyle = '-'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.figsize'] = [1.2,1]

data = zsubj_all[:,:,sensor].mean(axis=2) #(30,71) # 对选中的传感器求均值
d_mean = data.mean(axis=0)   # (71, )  # 计算每个时间点上的均值
d_sem = sem(data,axis=0)   # sem标准误差

def mark_win(times):    # 标记某个时间窗口的阴影区域
	shade_x = np.arange(times[0],times[-1])
	shade_y = d_mean[times[0]:times[-1]]
	text_x = (times[0]+times[-1])/2-0.5
	text_y = d_mean[int((times[0]+times[-1])/2)]/4
	return shade_x,shade_y,text_x,text_y
shade_x,shade_y,text_x,text_y = mark_win(times[0])

fig, ax = plt.subplots()
tstart = 0
tend = len(d_mean)
t = np.arange(tstart,tend)
plt.axvline(x=9,linewidth=0.75,linestyle=':',color='k',alpha=1)      #  画一条垂直虚线
plt.plot(t,d_mean,color='#5E65A2',linestyle=linestyle,linewidth=0.8,alpha=0.8) # 画均值曲线
plt.fill_between(t,d_mean-d_sem,d_mean+d_sem,alpha=0.1,facecolor='#5E65A2')   # 画均值 ± SEM 的阴影
# plt.fill_between(shade_x,-0.5,shade_y,facecolor='gray',alpha=0.3,interpolate=True)   #修改坐标
# ax.text(text_x,-0.35,"*",ha='center')   # 标记显著性符号 *
plt.xticks([0,19,39,59],[-100,100,300,500])  # 设置 X 轴刻度
plt.yticks([-0.3,0,0.3,0.6],[])   # # 设置 Y 轴刻度 (增加刻度数量)
ax.set_xlim((0,71.5))
ax.set_ylim((-0.5,0.7))        # 修改y坐标
ax.tick_params(axis='x',width=0.5,length=1.5,pad=0.5)
ax.tick_params(axis='y',width=0.5,length=1.5,pad=0.5)
ax.spines[['right','top']].set_visible(False)
ax.spines[['left','bottom']].set_linewidth(0.5)
plt.subplots_adjust(left=0.02,bottom=0.02,right=0.99,top=0.99)
# plt.savefig('/Users/elaine/Desktop/Results/timecourse_2.png')
plt.show()

#B43535 red
#5E65A2 blue

