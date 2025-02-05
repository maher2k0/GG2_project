import numpy as np
from attenuate import attenuate

def ct_detect(p, coeffs, depth, mas=10000):

	"""ct_detect returns detector photons for given material depths.
	y = ct_detect(p, coeffs, depth, mas) takes a source energy
	distribution photons (energies), a set of material linear attenuation
	coefficients coeffs (materials, energies), and a set of material depths
	in depth (materials, samples) and returns the detections at each sample
	in y (samples).

	mas defines the current-time-product which affects the noise distribution
	for the linear attenuation"""

	# check p for number of energies
	if type(p) != np.ndarray:
		p = np.array([p])
	if p.ndim > 1:
		raise ValueError('input p has more than one dimension')
	energies = len(p)

	# check coeffs is of (materials, energies)
	if type(coeffs) != np.ndarray:
		coeffs = np.array([coeffs]).reshape((1, 1))
	elif coeffs.ndim == 1:
		coeffs = coeffs.reshape((1, len(coeffs)))
	elif coeffs.ndim != 2:
		raise ValueError('input coeffs has more than two dimensions')
	if coeffs.shape[1] != energies:
		raise ValueError('input coeffs has different number of energies to input p')
	materials = coeffs.shape[0]

	# check depth is of (materials, samples)
	if type(depth) != np.ndarray:
		depth = np.array([depth]).reshape((1,1))
	elif depth.ndim == 1:
		if materials == 1:
			depth = depth.reshape(1, len(depth))
		else:
			depth = depth.reshape(len(depth), 1)
	elif depth.ndim != 2:
		raise ValueError('input depth has more than two dimensions')
	if depth.shape[0] != materials:
		raise ValueError('input depth has different number of materials to input coeffs')
	samples = depth.shape[1]

	# extend source photon array so it covers all samples
	detector_photons = np.zeros([energies, samples])
	for e in range(energies):
		detector_photons[e] = p[e]

	# calculate array of residual mev x samples for each material in turn
	for m in range(materials):
		detector_photons = attenuate(detector_photons, coeffs[m], depth[m])

	# sum this over energies
	detector_photons = np.sum(detector_photons, axis=0)

	# ????? how to determine constants needed ???????
	# test: with how much a constant will there be observable difference
	# model noise
	# calculate number of photons expected
	b_noise = True    # include background noise or not
	s_noise = False 	  # include scattering noise o not
	background = 0
	scattered = 0

	area = 0.01 	# scale ** 2  yes right
	detector_photons *= area*mas 
	#detector_photons = np.random.poisson(detector_photons/lambda_scale) * correction   #no of transmited photons follows poisson distribuion, lam value too much
	detector_photons = np.random.normal(detector_photons, detector_photons**0.5)


	# background noise
	if b_noise:
		background_level = 1e32		#per area				
		background = area*background_level
		background = np.random.normal(background, background**0.5)


	# scattering noise
	if s_noise:
		scatter_coef = 0.001
		scattered = np.sum(p)*area*mas*scatter_coef    #???
		scattered = np.random.normal(scattered, scattered**0.5)
	

	detector_photons = detector_photons + background + scattered

	# model noise	
	#detector_photons = np.random.normal(detector_photons, detector_photons**0.5)

	# minimum detection is one photon
	detector_photons = np.clip(detector_photons, 1, detector_photons.max())
	
	#print(detector_photons)
	
	return detector_photons

'''
ct_detect for every angle --> ct_scan to give sinogram --> calibrate --> ramp filer --> back_project to give reconstruction
'''