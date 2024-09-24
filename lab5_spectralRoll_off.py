# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:22:11 2024

@author: IT STUDENT
"""

import librosa
import matplotlib.pyplot as plt

# Load the audio signal (replace 'your_audio_file.wav' with your actual file)
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
waveform, sample_rate = librosa.load(filename)

# Compute the spectral roll-off
spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)

# Plot the spectral roll-off
plt.figure(figsize=(10, 4))
plt.semilogy(spectral_rolloff.T, label='Spectral Roll-off')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectral Roll-off')
plt.legend()
plt.show()
