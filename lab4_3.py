# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:10:20 2024

@author: IT STUDENT
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:24:55 2024

@author: IT STUDENT
"""

from numpy import empty, arange, exp, real, imag, pi
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import cv2

def dct2D(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

def idct2D(matrix):
    return idct(idct(matrix.T, norm='ortho').T, norm='ortho')

# Read the image
image = cv2.imread(r'C:\Users\IT STUDENT\MCC_202100424\girl_image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the DCT blocks array
dct_blocks = np.zeros_like(image, dtype=np.float64)

# Process each 8x8 block
for i in range(0, image.shape[0], 8):
    for j in range(0, image.shape[1], 8):
        block = image[i:i+8, j:j+8]
        dct_blocks[i:i+8, j:j+8] = dct2D(block)

# Print a sample 8x8 block and its DCT coefficients
sample_block = image[0:8, 0:8]
print("Sample 8x8 Block:\n", sample_block)
print("DCT Coefficients of the Sample Block:\n", dct2D(sample_block))

# Plot original image
plt.figure()
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Plot DCT coefficients of the first few 8x8 blocks
plt.figure()
plt.title('DCT Signals of 8x8 Blocks')
for i in range(0, 32, 8):
    for j in range(0, 32, 8):
        block = dct_blocks[i:i+8, j:j+8]
        plt.plot(block.flatten())
plt.xlabel('Flattened Index')
plt.ylabel('Coefficient Value')
plt.show()

