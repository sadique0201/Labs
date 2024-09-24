import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = (r'C:\Users\IT STUDENT\Desktop\sample-15s.wav')  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)

# Compute the Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

# Time vector for plotting
# Use the length of the spectral bandwidth feature to determine the time vector
frames = range(len(spectral_bandwidth[0]))
t = librosa.frames_to_time(frames, sr=sr)

# Plot the audio signal
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
plt.title('Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the Spectral Bandwidth
plt.subplot(2, 1, 2)
plt.plot(t, spectral_bandwidth[0], label='Spectral Bandwidth', color='orange')
plt.title('Spectral Bandwidth')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()

# Show plots
plt.show()
