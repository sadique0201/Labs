# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:18:41 2024

@author: IT STUDENT
"""

import librosa
import matplotlib.pyplot as plt

# Load the audio signal (replace 'your_audio_file.wav' with your actual file)
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
waveform, sample_rate = librosa.load(filename)

# Compute the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)

# Plot the spectral centroid
plt.figure(figsize=(10, 4))
plt.semilogy(spectral_centroids.T, label='Spectral Centroid')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectral Centroid')
plt.legend()
plt.show()
