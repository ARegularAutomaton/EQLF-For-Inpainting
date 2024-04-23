import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.storage import storage

def data_enumerator(path=storage.dataset_location(), train=True, batchsize=1, crop_size=(256,256), problem='inpaint'):
    if train:
        path = path + "train/"
    else:
        path = path +"test/"
    # Convert each image into a 3D tensor and each batch of images into a 4D tensor
    if problem == 'inpaint':
        transform_data = transforms.Compose([transforms.Resize(crop_size),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor()])
    elif problem == 'ct':
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                                            transforms.ToTensor(),
                                                            transforms.Grayscale()])
    dataset = datasets.ImageFolder(path, transform=transform_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return dataloader