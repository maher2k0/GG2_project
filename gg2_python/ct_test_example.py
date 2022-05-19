
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
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('100kVp, 3mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')

	# how to check whether these results are actually correct?

def resolution_real():
	# explain what this test is for
	#this test generates a phantom with only one tissue pixel at the center.
	#the ideal reconstructed image should show an impulse on line 127 and flat on all other lines.
	#In reconstructed images with real source, the width of the impulse at line 127 (*scale for real size) would be the resoultion of the scan, which is slightly bigger than one pixel

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(p[127,:], 'results', 'resolution_real_phantom_127')
	
	save_plot(y[127,:], 'results', 'resolution_real_plot_127')
	save_plot(y[128,:], 'results', 'resolution_real_plot_128')
	save_plot(y[129,:], 'results', 'resolution_real_plot_129')
	save_plot(y[130,:], 'results', 'resolution_real_plot_130')

	# how to check whether these results are actually correct?
	#The reconstructed image should show an impulse of certain width (~5 pixels = 0.5mm in this case).
	#Impulse are still visible on reconstructed image line 128 onwards, which spread out and die down eventually.

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
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# how to check whether these results are actually correct?


# Run the various tests
#print('Test 1')
#test_1()
#print('resolution_real')
#resolution_real()
print('resolution_ideal')
resolution_ideal()
print('Test 3')
test_3()
