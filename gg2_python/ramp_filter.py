import math
import numpy as np
import numpy.matlib

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
	m = int(2 ** m)
	print(m,angles,n)

	# apply filter to all angles
	print('Ramp filtering')
	fft_sinogram = np.fft.fft(sinogram,axis=1).T
	print(np.shape(fft_sinogram))
	padded = np.concatenate((fft_sinogram[:int(n/2)][:],np.zeros((n,angles))),axis=0)
	print(np.shape(padded))
	padded = np.concatenate((padded,fft_sinogram[int(n/2):][:]),axis=0).T	#n assumed even
	print(np.shape(padded))
	omega_0 = 1/(scale*n)
	omega_max = 1/(scale*2)
	ramp = np.zeros((m))
	for i in range(int(m/2)):
		ramp[i] = abs(omega_0*(i))/(2*np.pi) * np.cos(omega_0*(i)/omega_max*np.pi/2)**alpha
		i_negative = angles-i
		ramp[i_negative] = abs(omega_0*(i))/(2*np.pi) * np.cos(omega_0*(-i)/omega_max*np.pi/2)**alpha
	ramp[0] = ramp[1]/6

	filtered = padded*ramp
	return np.fft.fft(filtered)



	return sinogram