import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class CTDataset(Dataset):
    def __init__(self, image_folderpath, mask_folderpath):
        self.image_folderpath = image_folderpath
        self.mask_folderpath = mask_folderpath

        image_paths = []
        mask_paths = []

        for image_filename in os.listdir(image_folderpath): 
            image_path = os.path.join(image_folderpath, image_filename)
            image_paths.append(image_path)

        for mask_filename in os.listdir(mask_folderpath): 
            mask_path = os.path.join(mask_folderpath, mask_filename)
            mask_paths.append(mask_path)

        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = np.loadtxt(image_filepath)

        mask_filepath = self.mask_paths[idx]
        mask = np.loadtxt(mask_filepath)
        
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)   #shape (1,128,128)

        image, mask = torch.from_numpy(image.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))
        return image, mask
    
def display(display_list):
    plt.figure(figsize=(10,10))
    title = ['Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
    plt.show()

def show_dataset(datagen, num=1):
    iteration = iter(datagen)
    for i in range(0,num):
        image, mask = next(iteration)[0][0], next(iteration)[1][0]
        display([image, mask])

# class BrainSegmentationDataset(Dataset):
#     """Brain MRI dataset for FLAIR abnormality segmentation"""

#     in_channels = 1
#     out_channels = 1

#     def __init__(
#         self,
#         images_dir,
#         transform=None,
#         image_size=256,
#         subset="train",
#         random_sampling=True,
#         validation_cases=10,
#         seed=42,
#     ):
#         assert subset in ["all", "train", "validation"]

#         # read images
#         volumes = {}
#         masks = {}
#         print("reading {} images...".format(subset))
#         for (dirpath, dirnames, filenames) in os.walk(images_dir):
#             image_slices = []
#             mask_slices = []
#             for filename in sorted(
#                 filter(lambda f: ".tif" in f, filenames),
#                 key=lambda x: int(x.split(".")[-2].split("_")[4]),
#             ):
#                 filepath = os.path.join(dirpath, filename)
#                 if "mask" in filename:
#                     mask_slices.append(imread(filepath, as_gray=True))
#                 else:
#                     image_slices.append(imread(filepath))
#             if len(image_slices) > 0:
#                 patient_id = dirpath.split("/")[-1]
#                 volumes[patient_id] = np.array(image_slices[1:-1])
#                 masks[patient_id] = np.array(mask_slices[1:-1])

#         self.patients = sorted(volumes)

#         # select cases to subset
#         if not subset == "all":
#             random.seed(seed)
#             validation_patients = random.sample(self.patients, k=validation_cases)
#             if subset == "validation":
#                 self.patients = validation_patients
#             else:
#                 self.patients = sorted(
#                     list(set(self.patients).difference(validation_patients))
#                 )

#         print("preprocessing {} volumes...".format(subset))
#         # create list of tuples (volume, mask)
#         self.volumes = [(volumes[k], masks[k]) for k in self.patients]

#         print("cropping {} volumes...".format(subset))
#         # crop to smallest enclosing volume
#         self.volumes = [crop_sample(v) for v in self.volumes]

#         print("padding {} volumes...".format(subset))
#         # pad to square
#         self.volumes = [pad_sample(v) for v in self.volumes]

#         print("resizing {} volumes...".format(subset))
#         # resize
#         self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

#         print("normalizing {} volumes...".format(subset))
#         # normalize channel-wise
#         self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

#         # probabilities for sampling slices based on masks
#         self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
#         self.slice_weights = [
#             (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
#         ]

#         # add channel dimension to masks
#         self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

#         print("done creating {} dataset".format(subset))

#         # create global index for patient and slice (idx -> (p_idx, s_idx))
#         num_slices = [v.shape[0] for v, m in self.volumes]
#         self.patient_slice_index = list(
#             zip(
#                 sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
#                 sum([list(range(x)) for x in num_slices], []),
#             )
#         )

#         self.random_sampling = random_sampling

#         self.transform = transform

#     def __len__(self):
#         return len(self.patient_slice_index)

#     def __getitem__(self, idx):
#         patient = self.patient_slice_index[idx][0]
#         slice_n = self.patient_slice_index[idx][1]

#         if self.random_sampling:
#             patient = np.random.randint(len(self.volumes))
#             slice_n = np.random.choice(
#                 range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
#             )

#         v, m = self.volumes[patient]
#         image = v[slice_n]
#         mask = m[slice_n]

#         if self.transform is not None:
#             image, mask = self.transform((image, mask))

#         # fix dimensions (C, H, W)
#         image = image.transpose(2, 0, 1)
#         mask = mask.transpose(2, 0, 1)

#         image_tensor = torch.from_numpy(image.astype(np.float32))
#         mask_tensor = torch.from_numpy(mask.astype(np.float32))

#         # return tensors
#         return image_tensor, mask_tensor