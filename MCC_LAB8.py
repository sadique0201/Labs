import numpy as np
import cv2
from scipy.fftpack import dct, idct

# Convert image to YCbCr color space
def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# Apply DCT to an 8x8 block
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Quantization matrix for luminance
quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Quantize DCT coefficients
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

# Example usage
image = cv2.imread(r'D:\Prog Files\LABS\frames\frame_0196.jpg')
ycbcr_image = rgb_to_ycbcr(image)
height, width, _ = ycbcr_image.shape

# Process each 8x8 block
for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = ycbcr_image[i:i+8, j:j+8, 0]  # Y channel
        dct_block = apply_dct(block)
        quantized_block = quantize(dct_block, quant_matrix)
        # Further steps like entropy coding would follow here
