import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
class ImageFolderCar(ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderCar, self).__getitem__(index)
        path, _= self.samples[index]
        return sample, target, path

if __name__ == '__main__':
    file_loader = ImageFolderCar('/home/khanhhh/pCloudDrive/My Pictures/')
    for img, target, path in file_loader:
        print(path)