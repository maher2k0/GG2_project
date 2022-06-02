import argparse
import json
import os


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CTDataset
#from logger import Logger
from loss import DiceLoss
from unet import UNet
#from utils import log_images, dsc

def main():

    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')

    root = os.getcwd()
    mask_folder_path = os.path.join(root, 'data\\train\\mask')
    image_folder_path = os.path.join(root, 'data\\train\\image')
    LR = 0.00001
    EPOCH = 500

    CT_dataset = CTDataset(image_folder_path, mask_folder_path)

    loader_train, loader_valid = data_loaders(CT_dataset, CT_dataset)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=1, out_channels=1)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=LR)

    #logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(EPOCH), total=EPOCH):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                                                   
                                # logger.image_list_summary(
                                #     tag,
                                #     log_images(x, y_true, y_pred)[:num_images],
                                #     step,)
                              

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    #log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                #log_loss_summary(logger, loss_valid, step, prefix="val_")
                
                # mean_dsc = np.mean(
                #     dsc_per_volume(
                #         validation_pred,
                #         validation_true,
                #         loader_valid.dataset.patient_slice_index,
                #     )
                # )
                # logger.scalar_summary("val_dsc", mean_dsc, step)
                # if mean_dsc > best_validation_dsc:
                #     best_validation_dsc = mean_dsc
                #     torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    #print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    return unet

def data_loaders(dataset_train, dataset_valid):

    loader_train = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=0,
                    
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=16,
        drop_last=False,
        num_workers=0,

    )

    return loader_train, loader_valid


def datasets(image_folder_path, mask_folder_path):
    train = CTDataset(image_folder_path, mask_folder_path)

    valid = train = CTDataset(image_folder_path, mask_folder_path)
    
    return train, valid

'''
def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list

'''



def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)









if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    PATH = os.path.join(root, 'unet\\model\\epoch200_test_model.pth')
    print(PATH)
    unet = main()
    
    torch.save(unet, PATH)