
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


dataset = datasets.CIFAR10(
    root='cifar10',
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True)

