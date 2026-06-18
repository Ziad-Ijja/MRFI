import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import os


tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
])

testset = None


def get_testset(folder='~/dataset/val'):
    global testset
    folder = os.path.expanduser(folder)
    if testset is None:
        testset = datasets.ImageFolder(folder, tf)
    return testset


def make_testloader(size=None, folder='~/dataset/val', isRandom=False, **kwargs):
    testset = get_testset(folder)

    if size is None:
        return data.DataLoader(testset, **kwargs)

    if isRandom:
        idx = random.sample(range(len(testset)), size)
    else:
        idx = range(size)

    subset = data.Subset(testset, idx)
    return data.DataLoader(subset, **kwargs)