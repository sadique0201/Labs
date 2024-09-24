# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:21:48 2024

@author: IT STUDENT
"""

from numpy import empty,arange,exp,real,imag,pi
import numpy as np
from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import cv2
 

 
def DCT(frequency):
    M = len(frequency)
    dct_freq = np.zeros(M)
    for u in range(M):
        summation = 0
        for m in range(M):
            summation += np.cos((2*m + 1)*u*np.pi / (2*M)) * frequency[m]
        dct_freq[u] = (2 * 1 / np.sqrt(M)) * summation
    return dct_freq
 

 
N = 256
x = np.linspace(0, 10, N)
signal = np.sin(x)

plt.figure()
plt.title("Original Signal")
plt.xlabel("time")
plt.plot(signal)
plt.show()

def IDCT(frequency):
    M = len(frequency)
    idct_freq = np.zeros(M)
    summation = 0
    for m in range(M):
        summation += np.cos((2*m + 1)*m*np.pi / (2*M)) * frequency[m]
        idct_freq[m] = (2 * 1 / np.sqrt(M)) * summation
    return idct_freq
 

 
dct_coefficients = DCT(signal)

plt.figure()
plt.title("DCT Coefficients")
plt.plot(dct_coefficients)
plt.xlabel("time")
plt.show()

reconstructed_signal = idct(dct_coefficients)

plt.figure()
plt.title("Reconstructed Signal")
plt.plot(reconstructed_signal)
plt.show()