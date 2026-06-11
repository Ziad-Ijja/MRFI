import torch
from torch.nn.functional import normalize
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random


normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

tf=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

testset = None

def get_testset(folder = '~/dataset/val'):
    import os
    folder = os.path.expanduser(folder)
    global testset
    if testset is None:
        testset = datasets.ImageFolder(folder, tf)
    return testset

def make_testloader(size = None, folder = '~/dataset/val', isRandom=False, **kwargs):
    testset = get_testset(folder)
    if size is None:
        return data.DataLoader(testset, **kwargs)
    
    if isRandom:
        idx = random.sample(range(len(testset)), size)  
        subset = data.Subset(testset, idx)
        return data.DataLoader(subset, **kwargs)
    else:
        subset = data.Subset(testset, range(size))
        return data.DataLoader(subset, **kwargs)
