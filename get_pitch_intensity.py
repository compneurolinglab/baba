import os, sys
import parselmouth
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import spectrogram as sg
from scipy.signal import hilbert
from scipy.io import wavfile

# 从音频文件（.wav）中提取音高（f0）和强度（intensity），然后以 10ms 为时间间隔插值，并保存为 CSV 文件。
DIR = '/Volumes/T7_Shield/baba/praat/praat/'
os.chdir(DIR)

# get f0
snd = parselmouth.Sound('baba_audio.wav')
pitch = snd.to_pitch()
f0 = pitch.selected_array['frequency'].ravel()
print(len(f0))
length = 151900

# f0[np.isnan(f0)] = 0  # 如果 f0 为 NaN，将其设置为 0（或者其他值）
# f0[f0 == 0] = np.nan  # 将 f0 为 0 的值设置为 NaN
# # 对 f0 进行插值，填补 NaN 值
# f0_interp = pd.Series(f0)
# f0_interp.interpolate(method='linear', inplace=True)  # 用线性插值填充 NaN 值
# f0 = f0_interp.values  # 转回 numpy 数组

#get intensity
intensity = snd.to_intensity()
intensity_values = intensity.values.flatten()

# set timepoints (f0做一个插值)
original_time = np.linspace(0,snd.xmax,len(intensity_values))
time_10ms = np.arange(0,snd.xmax,0.01)
intensity1 = np.interp(time_10ms,original_time,intensity_values)[:length]

acoustics = np.column_stack((time_10ms[:length],f0[:length],intensity1))
df = pd.DataFrame(acoustics,columns=['time','f0','intensity'])
df.to_csv('/Users/elaine/Desktop/wav_acoustics.csv',index=False)
