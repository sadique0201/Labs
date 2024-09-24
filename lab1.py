# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:34:56 2024

@author: IT STUDENT
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import sounddevice as sd

# Load an example audio file
audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path)

# Compute the spectrogram
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# Frequency Masking
def frequency_mask(S_db, num_masks=1, freq_masking_max_percentage=0.15):
    S_db = S_db.copy()
    num_freqs = S_db.shape[0]

    for _ in range(num_masks):
        mask_percentage = random.uniform(0.0, freq_masking_max_percentage)
        mask_band = int(mask_percentage * num_freqs)
        start_freq = random.randint(0, num_freqs - mask_band)

        S_db[start_freq:start_freq + mask_band, :] = 0

    return S_db

# Temporal Masking
def temporal_mask(S_db, num_masks=1, time_masking_max_percentage=0.15):
    S_db = S_db.copy()
    num_times = S_db.shape[1]

    for _ in range(num_masks):
        mask_percentage = random.uniform(0.0, time_masking_max_percentage)
        mask_band = int(mask_percentage * num_times)
        start_time = random.randint(0, num_times - mask_band)

        S_db[:, start_time:start_time + mask_band] = 0

    return S_db

# Apply frequency and temporal masking
S_db_freq_masked = frequency_mask(S_db, num_masks=2)
S_db_temp_masked = temporal_mask(S_db, num_masks=2)

# Plot original and masked spectrograms
plt.figure(figsize=(12, 8))

# Original spectrogram
plt.subplot(3, 1, 1)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Spectrogram')

# Frequency masked spectrogram
plt.subplot(3, 1, 2)
librosa.display.specshow(S_db_freq_masked, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Frequency Masked Spectrogram')
print("playing original audio")
sd.play(y,sr)
sd.wait()

# Temporal masked spectrogram
plt.subplot(3, 1, 3)
librosa.display.specshow(S_db_temp_masked, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Temporal Masked Spectrogram')

plt.tight_layout()
plt.show()