from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from torchvision import models
import torchvision
import pandas as pd
import numpy as np
import torch

cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])

squeeze = models.squeezenet1_1
