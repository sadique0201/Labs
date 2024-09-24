# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:13:32 2024

@author: IT STUDENT
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio signal (replace 'your_audio_file.wav' with your actual file)
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
waveform, sample_rate = librosa.load(filename)

# Compute the Mel spectrogram
melspectrum = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, hop_length=512, window='hann', n_mels=256)

# Convert to decibels
melspectrum_db = librosa.power_to_db(melspectrum, ref=np.max)

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(melspectrum_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
