import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	n = np.shape(reconstruction)[1]
	print(np.shape(reconstruction))
	sinogram_w = ct_detect(p,material.coeff('Water'),1)	#one single water pixel scanned
	
	

	# put this through the same calibration process as the normal CT data
	I0E = ct_detect(p, material.coeff('Air'), 1)
	mu_w = -np.log(sinogram_w/I0E) 

	# use result to convert to hounsfield units
	# limit minimum to -1024, which is normal for CT data.
	reconstruction_hu = 1000*(reconstruction-mu_w)/mu_w
	reconstruction_hu[reconstruction_hu < -1024] = -1024

	return reconstruction_hu