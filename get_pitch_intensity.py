import os, sys
import parselmouth
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import spectrogram as sg
from scipy.signal import hilbert
from scipy.io import wavfile

DIR = '/scratch/ResearchGroups/lt_jixingli/baba'
os.chdir(DIR)

snd = parselmouth.Sound('baba_audio.wav')
pitch = snd.to_pitch()
f0 = pitch.selected_array['frequency'].ravel()
print(len(f0))
length = 151900

intensity = snd.to_intensity()
intensity_values = intensity.values.flatten()

original_time = np.linspace(0,snd.xmax,len(intensity_values))
time_10ms = np.arange(0,snd.xmax,0.01)
intensity1 = np.interp(time_10ms,original_time,intensity_values)[:length]

acoustics = np.column_stack((time_10ms[:length],f0[:length],intensity1))
df = pd.DataFrame(acoustics,columns=['time','f0','intensity'])
df.to_csv('wav_acoustics.csv',index=False)
