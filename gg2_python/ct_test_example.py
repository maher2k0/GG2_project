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

def geometry_check():
	# creates a phantom and uses the scan_and_reconstruct function to
	# simulate CT scan and reconstruct image from sinogram.
	# Testing reconstructed geometry

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('100kVp, 3mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_draw(y, 'results/geometry_check', 'geometry_check_image')
	save_draw(p, 'results/geometry_check', 'geometry_check_phantom')

	# To check if results are correct, visually inspect the phantom and reconstructed image
	# It's expected that the two images have the same geometry
	# They will however have different data levels since the phantom is constructed
	# from a list of indeces related to the material type whereas the reconstructed
	# image is constructed from a list of linear attenuations


def implant_noise_test(phantom = 3, mvp = 'high'):
	# explain what this test is for
	# This test is an extension off test 1, which checks whether the reconstruction 
	# matches phantom with fake photon source
	# this test checks noise patterns produced by dense implants under fake sources of 
	# different types (ideal or normal)

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, phantom)
	if mvp == 'high':
		energy = 0.2
		
	else:
		energy = 0.08
	s_ideal = fake_source(source.mev, energy, method='ideal')
	s_normal = fake_source(source.mev, energy, method='normal')

	# save some meaningful results
	y_ideal = scan_and_reconstruct(s_ideal, material, p, 0.1, 256)
	y_normal = scan_and_reconstruct(s_normal, material, p, 0.1, 256)
	save_draw(p, 'results/implant_noise_test', 'phantom ' 
			+ str(phantom), caxis = [0, p.max()])
	save_draw(y_ideal, 'results/implant_noise_test', 'phantom ' 
			+ str(phantom) + ' test ' + mvp + ' energy ideal source', caxis = [0, y_ideal.max()/4])
	save_draw(y_normal, 'results/implant_noise_test', 'phantom ' 
			+ str(phantom) + ' test ' + mvp + ' energy normal source', caxis = [0, y_normal.max()/4])


	# how to check whether these results are actually correct?
	# ideal source gives a single energy at peak value while normal source gives a distribution of energy around peak. 
	# It is expected that by using ideal source, the reconstruction will have less noise. This is verified by the graph.
	# in normal source graph, there is significant noise especially around dense implants such that the boundary in blurred.
	# The noise is significantly reduced in ideal source plot.


def resolution(source = 'real'):
	# explain what this test is for
	#this test generates a phantom with only one tissue pixel at the center.
	#the ideal reconstructed image should show an impulse on line 127 and flat on all other lines.
	#In reconstructed images with real source, the width of the impulse at line 127 (*scale for real size) would be the resoultion of the scan, which is slightly bigger than one pixel

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	if source == 'real':
		s = source.photon('80kVp, 1mm Al')
	elif source == 'ideal':
		s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 4,'ideal')
	elif source == 'normal':
		s = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 4,'normal')

	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(p[127,:], 'results/resolution', str(source) + '_resolution_phantom_127')
	
	save_plot(y[127,:], 'results/resolution', str(source) + '_resolution_plot_127')
	save_plot(y[128,:], 'results/resolution', str(source) + '_resolution_plot_128')
	save_plot(y[129,:], 'results/resolution', str(source) + '_resolution_plot_129')

	# how to check whether these results are actually correct?
	#The reconstructed image should show an impulse of certain width (~5 pixels = 0.5mm in this case).
	#Impulse are still visible on reconstructed image line 128 onwards, which spread out and die down eventually.
	#

def attenucation_softtissue():
	# This test calculates the mean attenuation of a section of the  reconstruced image
	# that is dominantly soft tissue. 

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/attenuation_softtissue_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# The mean attenuation value should be around 0.203963689535233cm^-1
	# which is the attenuation coefficient of soft tissue at 0.7*0.1MeV


def test_attenuate_fn():
    # testing the attenuate function
    # arbitrary list of attenuation coefficients, depths and energies are passed thorugh the attenuate function
	# assertion used to compare the theoretical values and the values obtained from the function

	coeffs = np.linspace(0.001,100,100)
	depths = np.linspace(0.1,10,100)
	energies=np.linspace(1,1000,100)
	res_photons_using_function = []
	for i in range(len(energies)):
		res_photons_using_function.append((attenuate(energies[i],coeffs[i],depths[i]))[0][0])
	res_photons = [energies*np.exp(-coeffs*depths)]

	assert(np.round_(res_photons, decimals = 5) == np.round_(res_photons_using_function, decimals = 5) ).all()
	print('test passed')
	return 0

    		
def attenuation_all_materials():
	# this test is similar to attenuation_softtissue, but with all materials rather than only soft tissue
	# compare the reconstructed and real attenuation coefficients
	#the result corresponds well with the datasheet values for most tissue materials, but is quite different for some dense metals
	
	recons = []
	real = []
	for name in material.name:
		# create a phantom and reconstruction
		s = fake_source(source.mev, 0.1, method='ideal')
		p = ct_phantom(material.name, 256//2, 1, metal=name)
		y = scan_and_reconstruct(s, material, p, 0.1, 256//2)

		# real attenuation coefficients from datasheet
		'''
		real =[0.0002118, 0.178483174, 0.20396369, 0.193529618, 0.19368997, 0.204619413, 
			   0.502343497,2.472408523,7.963246616, 4.818874719, 6.501935809, 0.284726852,
				9.115561613,5.443009699, 0.631764215, 9.621026538, 6.391060667, 6.491441387, 0.217277511]
		'''
		real.append(material.coeff(name)[np.argmax(s)])

		# get attenuation coefficients in reconstructions
		mean = np.mean(y[64//2:192//2, 64//2:192//2])
		recons.append(mean)

	plt.plot(material.name, real, label ='real attenuation coef')
	plt.plot(material.name, recons, label = 'reconstructed attenuation coef')
	plt.xticks(rotation=90)
	plt.legend()
	plt.savefig('results/attenuation_all_materials.png')
	plt.show()


def attenucation_energies(mat = 'Soft Tissue'):
	# this test is similar to attenucation_softtissue, but with different energies rather than only 0.1
	# compare the reconstructed and real attenuation coefficients
	# reconstructed attenuation coefficient should closely track the datasheet values at corresponding energies

	# NOTE: some materials gives very small results. Unsure why
	energies = source.mev[20:150][::8]
	recons = []
	real = []
	p = ct_phantom(material.name, 256//2, 1, metal=mat)
	for energy in energies:
		# create a phantom and reconstruction
		s = fake_source(source.mev, energy/0.7, method='ideal')
		y = scan_and_reconstruct(s, material, p, 0.1, 256//2)

		# real attenuation coefficients from datasheet
		material.coeff(mat)[np.argmax(s)]
		real.append(material.coeff(mat)[np.argmax(s)])

		# get attenuation coefficients in reconstructions
		mean = np.mean(y[64//2:192//2, 64//2:192//2])
		recons.append(mean)

	plt.plot(energies, real, label ='real attenuation coef')
	plt.plot(energies, recons, label = 'reconstructed attenuation coef')
	plt.xticks(rotation=90)
	plt.ylim((0,max(max(recons), max(real))*1.2))
	plt.xlabel('energy of source')
	plt.ylabel('attenuation coef')
	plt.legend()
	plt.title('check attenuation coefficient across ideal source with different energy, '+ mat)
	plt.savefig('results/attenucation_energies_'+mat+'.png')
	plt.show()


def attenucation_scales():
	# this test is similar to attenucation_softtissue, but with different scales rather than only 0.1
	# compare the reconstructed and real attenuation coefficients
	#reconstructed attenuation coefficient should be invariant to scale of the phantom

	scales = np.linspace(0.02,0.2,10)
	recons = []
	real = []
	
	for scale in scales:
		p = ct_phantom(material.name, 256//2, 1)
		# create a phantom and reconstruction
		s = fake_source(source.mev, 0.1, method='ideal')
		y = scan_and_reconstruct(s, material, p, scale, 256//2)

		# real attenuation coefficients from datasheet
		material.coeff('Soft Tissue')[np.argmax(s)]
		real.append(material.coeff('Soft Tissue')[np.argmax(s)])

		# get attenuation coefficients in reconstructions
		mean = np.mean(y[64//2:192//2, 64//2:192//2])
		recons.append(mean)

	plt.plot(scales, real, label ='real attenuation coef')
	plt.plot(scales, recons, label = 'reconstructed attenuation coef')
	plt.xticks(rotation=90)
	plt.ylim((0,max(max(recons), max(real))*1.2))
	plt.xlabel('scale')
	plt.ylabel('attenuation coef')
	plt.legend()
	plt.title('check attenuation coefficient across ideal source with different scale')
	plt.savefig('results/attenucation_energies.png')
	plt.show()


# Run the various tests
print('''Tests avaliable:
1. Geometry check()
2. implant noise test (phantom = 3/4/5/6/7, mvp = high/low)
3. resolution test (source = real/ideal/normal)
4. attenucation, single point check through reconstruction of a simple soft tissue phantom ()
5. attenuation, through ideal relationship at list of depths ()
6. attenuation, reconstruction of simple circular phantom of all materials ()
7. attenuation, with a list of energies(materials)
8. atteunation, with a list of scales()
''')
testcode = input('Please choose the test to be conducted: ')

testcode = testcode.split(' ')
if testcode[0] == '1':
	geometry_check()
if testcode[0] == '2':
	phantom = 3
	mvp = 'high'
	if len(testcode) >1 :
		phantom = int(testcode[1])
	if len(testcode) >2 :
		mvp = testcode[2]
	implant_noise_test(phantom, mvp)
if testcode[0] == '3':
	source = 'real'
	if len(testcode) >1:
		source = testcode[1]
	resolution(source)
if testcode[0] == '4':
	attenucation_softtissue()
if testcode[0] == '5':
	test_attenuate_fn()
if testcode[0] == '6':
	attenuation_all_materials()
if testcode[0] == '7':
	mat = 'Soft Tissue'
	if len(testcode) >1 :
		mat = testcode[1]
	attenucation_energies(mat)
if testcode[0] == '8':
	attenucation_scales()
