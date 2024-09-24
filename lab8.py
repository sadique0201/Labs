import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

# JPEG quantization matrix
QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

# Function to perform block-wise DCT on an 8x8 block
def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Function to perform block-wise inverse DCT on an 8x8 block
def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Function to apply quantization to the DCT coefficients
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

# Function to apply inverse quantization
def dequantize(block, quant_matrix):
    return block * quant_matrix

# Function to process an image block by block
def process_image(image, block_size=8):
    height, width = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)

    # Process image in 8x8 blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            
            # Perform DCT
            dct_block = dct_2d(block)

            # Quantize the DCT coefficients
            quant_block = quantize(dct_block, QUANTIZATION_MATRIX)

            # Dequantize (for reconstruction)
            dequant_block = dequantize(quant_block, QUANTIZATION_MATRIX)

            # Perform inverse DCT
            idct_block = idct_2d(dequant_block)

            # Save the block to the compressed image
            compressed_image[i:i+block_size, j:j+block_size] = idct_block

    # Clip values to valid range and convert to uint8
    compressed_image = np.clip(compressed_image, 0, 255)
    return compressed_image.astype(np.uint8)

# Main function to compress and display the image
def jpeg_compress(image_path):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image = np.array(image, dtype=np.float32)

    # Get original image dimensions
    height, width = image.shape

    # Calculate padding to make dimensions divisible by 8
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    # Pad the image with zeros if needed
    if pad_height != 0 or pad_width != 0:
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Process the image using DCT and quantization
    compressed_image = process_image(image)

    # Remove the padding after processing
    compressed_image = compressed_image[:height, :width]

    # Convert processed image to PIL format for display
    compressed_image_pil = Image.fromarray(compressed_image)

    # Display original and compressed images
    image_pil = Image.fromarray(image[:height, :width].astype(np.uint8))  # Original image before padding
    image_pil.show(title="Original Image")
    compressed_image_pil.show(title="Compressed Image")


# Path to your image file
image_path = (r'C:\Users\IT STUDENT\MCC_202100424\OIP.jpg')

# Run the JPEG compression
jpeg_compress(image_path)