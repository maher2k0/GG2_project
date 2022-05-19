
# these are the imports you are likely to need
from cv2 import mean
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# creates a phantom and uses the scan_and_reconstruct function to
	# simulate CT scan and reconstruct image from sinogram.
	# Testing reconstructed geometry

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('100kVp, 3mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')

	# To check if results are correct, visually inspect the phantom and reconstructed image
	# It's expected that the two images have the same geometry
	# They will however have different data levels since the phantom is constructed
	# from a list of indeces related to the material type whereas the reconstructed
	# image is constructed from a list of linear attenuations


def implant_noise_test(phantom = 3, mvp = 'high', method = 'ideal'):
	# explain what this test is for
	# This test is an extension off test 1, which checks whether the reconstruction 
	# matches phantom with fake photon source
	# this test checks noise patterns produced by dense implants under fake sources of 
	# different types (ideal or normal)

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, phantom)
	if mvp == 'high':
		s = fake_source(source.mev, 0.2, method=method)
	else:
		s = fake_source(source.mev, 0.08, method=method)

	# save some meaningful results
	y = scan_and_reconstruct(s, material, p, 0.1, 256)
	save_draw(p, 'results/implant_noise_test', 'phantom ' 
			+ str(phantom), caxis = [0, p.max()])
	save_draw(y, 'results/implant_noise_test', 'phantom ' 
			+ str(phantom) + ' test ' + mvp + ' energy ' + method + ' source', caxis = [0, y.max()/4])


	# how to check whether these results are actually correct?
	# ideal source gives a single energy at peak value while normal source gives a distribution of energy around peak. 
	# It is expected that by using ideal source, the reconstruction will have less noise. This is verified by the graph.
	# in normal source graph, there is significant noise especially around dense implants such that the boundary in blurred.
	# The noise is significantly reduced in ideal source plot.


def resolution_real():
	# explain what this test is for
	#this test generates a phantom with only one tissue pixel at the center.
	#the ideal reconstructed image should show an impulse on line 127 and flat on all other lines.
	#In reconstructed images with real source, the width of the impulse at line 127 (*scale for real size) would be the resoultion of the scan, which is slightly bigger than one pixel

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256, caxis = [0,6])

	# save some meaningful results
	save_plot(p[127,:], 'results', 'resolution_real_phantom_127')
	
	save_plot(y[127,:], 'results', 'resolution_real_plot_127')
	save_plot(y[128,:], 'results', 'resolution_real_plot_128')
	save_plot(y[129,:], 'results', 'resolution_real_plot_129')
	save_plot(y[130,:], 'results', 'resolution_real_plot_130')

	# how to check whether these results are actually correct?
	#The reconstructed image should show an impulse of certain width (~5 pixels = 0.5mm in this case).
	#Impulse are still visible on reconstructed image line 128 onwards, which spread out and die down eventually.

# def test_2():
	#test_2 is replaced by resolution_real() above which includes and extends from test_2
#     	# returns impulse response of the back projection

# 	# work out what the initial conditions should be
# 	p = ct_phantom(material.name, 256, 2)
# 	s = source.photon('80kVp, 1mm Al')
# 	y = scan_and_reconstruct(s, material, p, 0.01, 256)

# 	# save some meaningful results
# 	save_plot(y[128,:], 'results', 'test_2_plot')

# 	# Expecting a sharp pulse

def resolution_ideal():
	# explain what this test is for
	#this test generates a phantom with only one tissue pixel at the center.
	#the ideal reconstructed image should show an impulse on line 127 and flat on all other lines.
	#In reconstructed images with real source, the width of the impulse at line 127 (*scale for real size) would be the resoultion of the scan, which is slightly bigger than one pixel
	#this test used a fake source instead of a real one, the result turned out to be the same

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 4,'ideal')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(p[127,:], 'results', 'resolution_ideal_phantom_127')
	
	save_plot(y[127,:], 'results', 'resolution_ideal_plot_127')
	save_plot(y[128,:], 'results', 'resolution_ideal_plot_128')
	save_plot(y[129,:], 'results', 'resolution_ideal_plot_129')
	save_plot(y[130,:], 'results', 'resolution_ideal_plot_130')

	# how to check whether these results are actually correct?
	#The reconstructed image should show an impulse of certain width (~5 pixels = 0.5mm in this case).
	#Impulse are still visible on reconstructed image line 128 onwards, which spread out and die down eventually.

def test_3():
	# This test calculates the mean attenuation of a section of the  reconstruced image
	# that is dominantly soft tissue. 

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256, alpha = 0)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# The mean attenuation value should be around 0.203963689535233
	# which is the attenuation coefficient of soft tissue at 0.1MeV * 0.7 = 0.07MeV
	# our results is 0.19002429019883954, and stays around this value after varying n and scale 


def test_4():
	# this test is similar to test_3, but with all materials rather than only soft tissue
	# compare the reconstructed and real attenuation coefficients
	mat_names = ['Air',
				'Adipose',
				'Soft Tissue',
				'Breast Tissue',
				'Water',
				'Blood',
				'Bone',
				'Titanium',
				'Cobalt',
				'Chromium',
				'Iron',
				'Carbon',
				'Nickel',
				'Manganese',
				'Aluminium',
				'Copper',
				'Co-Cr',
				'Stainless Steel',
				'Acrylic']

	recons = []
	for name in mat_names:
		# create a phantom and reconstruction
		s = fake_source(source.mev, 0.1, method='ideal')
		p = ct_phantom(material.name, 256//2, 1, metal=name)
		y = scan_and_reconstruct(s, material, p, 0.1, 256//2)

		# real attenuation coefficients from datasheet
		real =[0.0002118, 0.178483174, 0.20396369, 0.193529618, 0.19368997, 0.204619413, 
			   0.502343497,2.472408523,7.963246616, 4.818874719, 6.501935809, 0.284726852,
				9.115561613,5.443009699, 0.631764215, 9.621026538, 6.391060667, 6.491441387, 0.217277511]

		# get attenuation coefficients in reconstructions
		mean = np.mean(y[64//2:192//2, 64//2:192//2])
		recons.append(mean)

	plt.plot(mat_names, real, label ='real attenuation coef')
	plt.plot(mat_names, recons, label = 'reconstructed attenuation coef')
	plt.xticks(rotation=90)
	plt.legend()
	plt.savefig('results/test_4.png')
	plt.show()


# Run the various tests
print('Test 1')
test_1()
# print('Test 2')
# test_2()
#print('resolution_real')
#resolution_real()
print('resolution_ideal')
resolution_ideal()
print('Test 3')
test_3()
implant_noise_test(phantom = 3, method = 'ideal')
implant_noise_test(phantom = 3, method = 'ideal')
test_4()



