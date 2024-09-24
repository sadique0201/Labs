# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:17:32 2024

@author: IT STUDENT
"""

import librosa
import matplotlib.pyplot as plt

# Load the audio signal (replace 'your_audio_file.wav' with your actual file)
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
waveform, sample_rate = librosa.load(filename)

# Compute the MFCCs
mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

# Visualize the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCCs')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficient')
plt.show()
