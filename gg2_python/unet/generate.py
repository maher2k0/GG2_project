import argparse
import json
import os

import pydicom
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# from dataset import BrainSegmentationDataset as Dataset
# from logger import Logger
# from loss import DiceLoss
# from transform import transforms
# from unet import UNet
# from utils import log_images, 



def read_dcm(folder_name):
    # return 3d array of hu unit, pixelspacing and slice thickness
    root = os.getcwd()
    folder_path = os.path.join(root, folder_name)
    fnames = []


    for file in os.listdir(folder_path):
        if file.endswith(".dcm"): 
            fname = os.path.join(folder_path, file)
            fnames.append(fname)
      

    # load the DICOM files
    files = []

    #print('glob: {}'.format(sys.argv[1]))
    for fname in fnames:
        #print("loading: {}".format(fname))
        files.append(pydicom.dcmread(fname))

    #print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)    # slices contain all files with this attr
        else:
            skipcount = skipcount + 1

    #print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness


    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)   # shape of one slice
    img_shape.append(len(slices))    # shape of one slice * no of slice
    print('model shape: ', img_shape)
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d


    return img3d, ps, ss

def plot_3d(img3d, ps, ss,):
    #
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    img_shape = img3d.shape
    plt.figure(figsize=(15, 15))
    
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, img_shape[2]//2])
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, img_shape[1]//2, :])
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[img_shape[0]//2, :, :].T)
    a3.set_aspect(cor_aspect)

    plt.colorbar()
    
    plt.show()
    
    return None

def cylinder_crop(img3d, ratio = 0.9, cut = 2):
    # cut = 2 means img3d[:, :, xxx] xxx will be cropped
    if cut == 2:
        length, width = img3d.shape[0], img3d.shape[1]
    elif cut == 1:
        length, width = img3d.shape[0], img3d.shape[2]
    elif cut == 0:
        length, width = img3d.shape[1], img3d.shape[2]

    radius = int(min(length, width)/2 * ratio) # radius of cropped circle
    xi, yi = np.meshgrid(np.arange(0,length,1) - (length/2) + 0.5, np.arange(0,width,1) - (width/2) + 0.5)
    img3d[np.where((xi ** 2 + yi ** 2) > radius**2)] = -1000 # -1000 is HU of air
        
    return img3d

def straight_crop(img3d, ratio = 0.9, cut = 2):
    n = int(img3d.shape[cut]*(1-ratio))
    if cut == 2: 
        img3d[:, :, :n] = -1000
    elif cut == 1: 
        img3d[:, :n, :] = -1000
    elif cut == 0: 
        img3d[:n, :, :] = -1000

    return img3d


img3d, ps, ss = read_dcm('recon_data_a')
img3d = cylinder_crop(img3d, ratio = 0.8)
img3d = straight_crop(img3d, ratio = 0.9)
print(img3d.shape)
'''
plot_3d(img3d, ps, ss,)
'''
HOUNSFIELD_MIN = -1024
HOUNSFIELD_MAX = 4096
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN


# normalise image
def normalise(img3d):
    img3d[img3d < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img3d[img3d > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img3d - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

img3d = normalise(img3d)
np.min(img3d), np.max(img3d), img3d.shape, type(img3d)


def generate_and_save_mask(img3d, mask_folder_path, img_folder_path):
    img_shape = img3d.shape
    x, y, z= img_shape
    mask3d = np.zeros(img_shape)
    threshold = 0.6
    for i in range(z):
        if i<= 58:
            slice = img3d[:, :, i]
            mask = slice > threshold
            mask[60:, 75:] = 0
            mask3d[:, :, i] = mask
        mask_fname = 'z_mask' + "{:03d}".format(i) + '.txt'
        img_fname = 'z_img' + "{:03d}".format(i) + '.txt'
        mask_save_path = os.path.join(mask_folder_path, mask_fname)
        img_save_path = os.path.join(img_folder_path, img_fname)

        np.savetxt(mask_save_path, mask, fmt='%d')
        np.savetxt(img_save_path, slice, fmt='%f')
    return mask3d


root = os.getcwd()
mask_folder_path = os.path.join(root, 'data\\train\\mask')
img_folder_path = os.path.join(root, 'data\\train\\image')

mask3d = generate_and_save_mask(img3d, mask_folder_path, img_folder_path)
