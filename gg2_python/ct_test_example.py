
# these are the imports you are likely to need
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
	save_draw(y, 'results', 'test_1_image', caxis =[0,6])
	save_draw(p, 'results', 'test_1_phantom')

	# To check if results are correct, visually inspect the phantom and reconstructed image
	# It's expected that the two images have the same geometry
	# They will however have different data levels since the phantom is constructed
	# from a list of indeces related to the material type whereas the reconstructed
	# image is constructed from a list of linear attenuations

def test_2():
	# returns impulse response of the back projection

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')

	# Expecting a sharp pulse

def test_3():
	# This test calculates the mean attenuation of a section of the  reconstruced image
	# that is dominantly soft tissue. 

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# The mean attenuation value should be around 0.179458cm^-1
	# which is the attenuation coefficient of soft tissue at 0.1MeV


# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
