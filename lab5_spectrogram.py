# -*- coding: utf-8 -*-
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio signal
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
waveform, sample_rate = librosa.load(filename)

# Compute the spectrogram
spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=512)

# Convert to decibels
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

