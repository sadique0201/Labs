# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:04:34 2024

@author: IT STUDENT
"""

import librosa
import matplotlib.pyplot as plt

# Load the audio signal
filename = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')
y, sr = librosa.load(filename, sr=None)

# Compute ZCR
zcrs = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)

# Print the ZCR values (one value per frame)
print(zcrs)

# Plot the ZCR over time
plt.figure(figsize=(10, 4))
plt.plot(zcrs[0])
plt.xlabel('Frame')
plt.ylabel('Zero Crossing Rate')
plt.title('Zero Crossing Rate vs. Frame')
plt.show()
