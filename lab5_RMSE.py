# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:57:04 2024

@author: IT STUDENT
"""

import librosa
import matplotlib.pyplot as plt

# Load the audio signal
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
y, sr = librosa.load(filename, sr=None)

# Compute RMSE
rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

# Print the RMSE values (one value per frame)
print(rmse)


# Plot the RMSE over time
plt.figure(figsize=(10, 4))
plt.plot(rmse[0])
plt.xlabel('Frame')
plt.ylabel('RMSE')
plt.title('RMSE vs. Frame')
plt.show()
