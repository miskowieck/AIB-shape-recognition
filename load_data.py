import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for transformation

import torch  # PyTorch package
import torchvision  # load datasets
import torchvision.transforms as transforms  # transform data
from torchvision.datasets import ImageFolder
import torch.nn as nn  # basic building block for neural neteorks
import torch.nn.functional as F  # import convolution functions like Relu
import torch.optim as optim  # optimzer

# python image library of range [0, 1]
# transform them to tensors of normalized range[-1, 1]
import imageio
import glob

from torch.utils.data import DataLoader


def load_data():

    # data = []
    # for im_path in glob.glob("C:/Users/Luki/Downloads/archive/shapes/circle/*.png"):
    #      im = imageio.imread(im_path)
    #      data.append(im)
    #
    #
    # for im_path in glob.glob("C:/Users/Luki/Downloads/archive/shapes/circle/*.png"):
    #     im = imageio.imread(im_path)
    #     print(im)
    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    transform = transforms.Compose(  # composing several transforms together
        [transforms.ToTensor(),  # to tensor object
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # mean = 0.5, std = 0.5

    data_path = 'C:\\Users\\Admin\\PycharmProjects\\AIB-shape-recognition\\data\\shapes'

    train_dataset = ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=1,
    )

    classes = ('circle', 'square', 'star', 'triangle')

    return train_loader, classes

    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=num_workers)
    #
    # # load test data
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=num_workers)

    # put 10 classes into a set

load_data()