import librosa
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(y, frame_length, hop_length):
    # Compute the squared energy of each frame
    energy = np.array([
        np.sum(np.square(y[i:i+frame_length]))
        for i in range(0, len(y), hop_length)
    ])
    return energy

# Load an example audio file from librosa
y, sr = librosa.load(librosa.ex('trumpet'))

# Parameters
frame_length = 2048  # Number of samples per frame
hop_length = 512     # Number of samples between each frame

# Compute energy
energy = compute_energy(y, frame_length, hop_length)

# Plot the energy
plt.figure(figsize=(14, 5))
plt.plot(energy)
plt.title('Energy of the audio signal')
plt.xlabel('Frame')
plt.ylabel('Energy')
plt.show()