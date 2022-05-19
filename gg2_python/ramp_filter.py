import math
import numpy as np
import numpy.matlib
from scipy.fftpack import fft
import matplotlib.pyplot as plt
def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)    #m is at least twich as long as n
	
	w = np.linspace(-np.pi/scale, np.pi/scale, m)
	#w = np.linspace(-1/scale, 1/scale, m)
	filter = abs(w)/(2*np.pi) 
	power_term = np.cos(w*np.pi/(2*np.pi/scale))**alpha
	filter = filter*power_term

	filter = np.concatenate((filter[int(m//2):], filter[0:int(m//2)]))

	
	#plt.plot(filter)
	#plt.show()

    # fft sinogram in the r direction, zero padding so that output sequence has length m
	sino_fft = np.fft.fft(sinogram, axis=1, n=m)
	
	print(sino_fft.shape, filter.shape)
    # filter by multiplying filter and sinogram in fourier domain
	filtered_sino = sino_fft * filter[np.newaxis, :]
    # Inverse fft, then trancate to reach original length
	sino = np.fft.ifft(filtered_sino, axis=1)[:, :n]

	return np.real(sino)    #elements in sino are complex, abs OR real????????????
	#return np.real(sino)
